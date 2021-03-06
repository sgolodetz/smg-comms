import socket
import threading

from typing import Callable, cast, Optional, Tuple

from smg.utility import PooledQueue

from ..base import AckMessage, CalibrationMessage, FrameHeaderMessage, FrameMessage, SocketUtil


class MappingClient:
    """A client that can be used to communicate with a remote mapping server."""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7851), *, timeout: int = 10,
                 frame_compressor: Optional[Callable[[FrameMessage], FrameMessage]] = None,
                 pool_empty_strategy: PooledQueue.EPoolEmptyStrategy = PooledQueue.PES_DISCARD):
        """
        Construct a mapping client.

        :param endpoint:            The server host and port, e.g. ("127.0.0.1", 7851).
        :param timeout:             The socket timeout to use (in seconds).
        :param frame_compressor:    An optional function to use to compress frames prior to transmission.
        :param pool_empty_strategy: The strategy to use when an attempt is made to send a frame message whilst the
                                    pool of frames associated with the frame message queue is empty.
        """
        self.__alive = False                         # type: bool
        self.__calib_msg = None                      # type: Optional[CalibrationMessage]
        self.__frame_compressor = frame_compressor   # type: Optional[Callable[[FrameMessage], FrameMessage]]
        self.__frame_message_queue = PooledQueue[FrameMessage](pool_empty_strategy)  # type: PooledQueue[FrameMessage]
        self.__message_sender_thread = None          # type: Optional[threading.Thread]
        self.__should_terminate = threading.Event()  # type: threading.Event

        try:
            self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # type: socket.SocketType
            self.__sock.connect(endpoint)
            self.__sock.settimeout(timeout)
            self.__alive = True
        except ConnectionRefusedError:
            raise RuntimeError("Error: Could not connect to the server")

    # DESTRUCTOR

    def __del__(self):
        """Destroy the client."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the client's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the client at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def send_calibration_message(self, calib_msg: CalibrationMessage) -> None:
        """
        Send a calibration message to the server.

        :param calib_msg:   The calibration message.
        """
        connection_ok = True  # type: bool

        # Send the message to the server.
        connection_ok = connection_ok and SocketUtil.write_message(self.__sock, calib_msg)

        # Wait for an acknowledgement (note that this is blocking, unless the connection fails).
        ack_msg = AckMessage()  # type: AckMessage
        connection_ok = connection_ok and SocketUtil.read_message(self.__sock, ack_msg)

        # If the meessage was successfully sent and acknowledged, save it, else throw.
        if connection_ok:
            self.__calib_msg = calib_msg
        else:
            raise RuntimeError("Error: Failed to send calibration message")

        # Initialise the frame message queue.
        capacity = 1  # type: int
        self.__frame_message_queue.initialise(capacity, lambda: FrameMessage(
            calib_msg.get_image_shapes(), calib_msg.get_uncompressed_image_byte_sizes()
        ))

        # Start the message sender thread.
        self.__message_sender_thread = threading.Thread(target=self.__run_message_sender)
        self.__message_sender_thread.start()

    def send_frame_message(self, frame_filler: Callable[[FrameMessage], None]) -> None:
        """
        Send a frame message to the server.

        .. note::
            In order to support different types of frame message, it is imperative that the client doesn't
            know anything about the contents of the messages being sent. We achieve this by making it call
            a callback function that fills in the contents of a message.

        :param frame_filler:    A callback function that should fill in the contents of a message.
        """
        with self.__frame_message_queue.begin_push(self.__should_terminate) as push_handler:
            elt = push_handler.get()  # type: Optional[FrameMessage]
            if elt is not None:
                msg = cast(FrameMessage, elt)  # type: FrameMessage
                frame_filler(msg)

    def terminate(self) -> None:
        """Tell the client to terminate."""
        if self.__alive:
            self.__should_terminate.set()
            if self.__message_sender_thread is not None:
                self.__message_sender_thread.join()
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False

    # PRIVATE METHODS

    def __run_message_sender(self) -> None:
        """Send frame messages from the message queue across to the server."""
        ack_msg = AckMessage()  # type: AckMessage

        connection_ok = True  # type: bool

        while connection_ok and not self.__should_terminate.is_set():
            # Try to read the first frame message from the queue (this will block until a message is available,
            # except when the termination flag is set, in which case it will return None).
            frame_msg = self.__frame_message_queue.peek(self.__should_terminate)  # type: Optional[FrameMessage]

            # If the termination flag is set, exit.
            if self.__should_terminate.is_set():
                break

            # If requested, compress the frame prior to transmission.
            compressed_frame_msg = frame_msg  # type: FrameMessage
            if self.__frame_compressor is not None:
                compressed_frame_msg = self.__frame_compressor(frame_msg)

            # Make the frame header message.
            header_msg = FrameHeaderMessage(self.__calib_msg.get_max_images())  # type: FrameHeaderMessage
            header_msg.set_image_byte_sizes(compressed_frame_msg.get_image_byte_sizes())
            header_msg.set_image_shapes(compressed_frame_msg.get_image_shapes())

            # First send the frame header message, then send the frame message, then wait for an acknowledgement
            # from the server. We chain all of these with 'and' so as to early out in case of failure.
            connection_ok = connection_ok and \
                SocketUtil.write_message(self.__sock, header_msg) and \
                SocketUtil.write_message(self.__sock, compressed_frame_msg) and \
                SocketUtil.read_message(self.__sock, ack_msg)

            # If the frame message was successfully sent, remove it from the queue. If not, set the termination flag.
            if connection_ok:
                self.__frame_message_queue.pop(self.__should_terminate)
            else:
                self.__should_terminate.set()
