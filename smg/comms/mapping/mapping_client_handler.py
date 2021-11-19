import numpy as np
import socket
import threading

from typing import Callable, cast, List, Optional, Tuple, TypeVar

from smg.utility import PooledQueue

from ..base import AckMessage, CalibrationMessage, FrameHeaderMessage, FrameMessage, Message, SocketUtil


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class MappingClientHandler:
    """Used to manage the connection to a mapping client."""

    # CONSTRUCTOR

    def __init__(self, client_id: int, sock: socket.SocketType, should_terminate: threading.Event, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None,
                 pool_empty_strategy: PooledQueue.EPoolEmptyStrategy = PooledQueue.PES_DISCARD):
        """
        Construct a mapping client handler.

        :param client_id:           The ID used by the server to refer to the client.
        :param sock:                The socket used to communicate with the client.
        :param should_terminate:    Whether or not the server should terminate (read-only, set within the server).
        :param frame_decompressor:  An optional function to use to decompress received frames.
        :param pool_empty_strategy: The strategy to use when a frame message is received whilst the pool of frames
                                    associated with the frame message queue is empty.
        """
        self.__calib_msg = None                         # type: Optional[CalibrationMessage]
        self.__client_id = client_id                    # type: int
        self.__connection_ok = True                     # type: bool
        self.__frame_decompressor = frame_decompressor  # type: Optional[Callable[[FrameMessage], FrameMessage]]
        self.__frame_message_queue = PooledQueue[FrameMessage](pool_empty_strategy)  # type: PooledQueue[FrameMessage]
        self.__lock = threading.Lock()                  # type: threading.Lock
        self.__newest_frame_msg = None                  # type: Optional[FrameMessage]
        self.__should_terminate = should_terminate      # type: threading.Event
        self.__sock = sock                              # type: socket.SocketType
        self.__thread = None                            # type: Optional[threading.Thread]

    # PUBLIC METHODS

    def get_client_id(self) -> int:
        """
        Get the ID used by the server to refer to the client.

        :return:    The ID used by the server to refer to the client.
        """
        return self.__client_id

    def get_frame(self, receiver: Callable[[FrameMessage], None]) -> None:
        """
        Get the oldest frame from the client that has not yet been processed.

        .. note::
            The concept of a 'frame receiver' is used to obviate the client handler from needing to know about
            the contents of frame messages. This way, the frame receiver needs to know how to handle the frame
            message that it's given, but the client handler can just forward it to the receiver without caring.

        :param receiver:    The frame receiver to which to pass the oldest frame from the client that has not
                            yet been processed.
        """
        with self.__lock:
            # Pass the first frame on the message queue to the frame receiver.
            receiver(self.__frame_message_queue.peek(self.__should_terminate))

            # Pop the frame that's just been read from the message queue.
            self.__frame_message_queue.pop(self.__should_terminate)

    def get_image_shapes(self) -> Optional[List[Tuple[int, int, int]]]:
        """
        Try to get the shapes of the images being produced by the different cameras being used.

        :return:    The shapes of the images being produced by the different cameras, if a calibration
                    message has been received from the client, or None otherwise.
        """
        return self.__calib_msg.get_image_shapes() if self.__calib_msg is not None else None

    def get_intrinsics(self) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Try to get the intrinsics of the different cameras being used.

        :return:    The intrinsics of the different cameras being used, as (fx, fy, cx, cy) tuples,
                    if a calibration message has been received from the client, or None otherwise.
        """
        return self.__calib_msg.get_intrinsics() if self.__calib_msg is not None else None

    def has_frames_now(self) -> bool:
        """
        Get whether or not the client is ready to yield a frame.

        :return:    True, if the client is ready to yield a frame, or False otherwise.
        """
        with self.__lock:
            return not self.__frame_message_queue.empty()

    def is_connection_ok(self) -> bool:
        """
        Get whether the connection is still ok (tracks whether or not the most recent read/write succeeded).

        :return:    True, if the connection is still ok, or False otherwise.
        """
        return self.__connection_ok

    def peek_newest_frame(self, receiver: Callable[[FrameMessage], None]) -> bool:
        """
        Peek at the newest frame received from the client (if any).

        .. note::
            The concept of a 'frame receiver' is used to obviate the server from needing to know about the contents
            of frame messages. This way, the frame receiver needs to know how to handle the frame message that it's
            given, but the server can just forward it to the receiver without caring.

        :param receiver:    The frame receiver to which to pass the newest frame from the client.
        :return:            True, if a newest frame existed and was passed to the receiver, or False otherwise.
        """
        with self.__lock:
            # If any frame has ever been received from the client, pass the newest frame to the frame receiver.
            if self.__newest_frame_msg is not None:
                receiver(self.__newest_frame_msg)
                return True
            else:
                return False

    def run_iter(self) -> None:
        """Run an iteration of the main loop for the client."""
        # Try to read a frame header message.
        header_msg = FrameHeaderMessage(self.__calib_msg.get_max_images())  # type: FrameHeaderMessage
        self.__connection_ok = SocketUtil.read_message(self.__sock, header_msg, self.__should_terminate)

        # If that succeeds:
        if self.__connection_ok:
            # Set up a frame message accordingly.
            image_shapes = header_msg.get_image_shapes()              # type: List[Tuple[int, int, int]]
            image_byte_sizes = header_msg.get_image_byte_sizes()      # type: List[int]
            frame_msg = FrameMessage(image_shapes, image_byte_sizes)  # type: FrameMessage

            # Try to read the contents of the frame message from the client.
            self.__connection_ok = SocketUtil.read_message(self.__sock, frame_msg, self.__should_terminate)

            # If that succeeds:
            if self.__connection_ok:
                # Decompress the frame as necessary.
                decompressed_frame_msg = frame_msg  # type: FrameMessage
                if self.__frame_decompressor is not None:
                    decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                # Save the decompressed frame as the newest one we have received. We do this so that we have a
                # record of the newest frame received (e.g. to serve peeks) even if the message queue empties.
                with self.__lock:
                    self.__newest_frame_msg = decompressed_frame_msg

                # Also push the decompressed frame onto the message queue.
                with self.__frame_message_queue.begin_push(self.__should_terminate) as push_handler:
                    elt = push_handler.get()  # type: Optional[FrameMessage]
                    if elt is not None:
                        msg = cast(FrameMessage, elt)  # type: FrameMessage
                        np.copyto(msg.get_data(), decompressed_frame_msg.get_data())

                # Send an acknowledgement to the client.
                self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def run_pre(self) -> None:
        """Run any code that should happen before the main loop for the client."""
        # Read a calibration message from the client.
        self.__calib_msg = CalibrationMessage()
        self.__connection_ok = SocketUtil.read_message(self.__sock, self.__calib_msg)

        # If the calibration message was successfully read:
        if self.__connection_ok:
            # Print the camera parameters out for debugging purposes.
            image_shapes = self.__calib_msg.get_image_shapes()  # type: List[Tuple[int, int, int]]
            intrinsics = self.__calib_msg.get_intrinsics()      # type: List[Tuple[float, float, float, float]]
            print(
                "Received camera parameters from client {}: {}, {}".format(self.__client_id, image_shapes, intrinsics)
            )

            # Initialise the frame message queue.
            capacity = 5  # type: int
            self.__frame_message_queue.initialise(capacity, lambda: FrameMessage(
                self.__calib_msg.get_image_shapes(), self.__calib_msg.get_uncompressed_image_byte_sizes()
            ))

            # Signal to the client that the server is ready.
            self.__connection_ok = SocketUtil.write_message(self.__sock, AckMessage())

    def set_thread(self, thread: threading.Thread) -> None:
        """
        Set the thread that manages communication with the client.

        :param thread:  The thread that manages communication with the client.
        """
        self.__thread = thread
