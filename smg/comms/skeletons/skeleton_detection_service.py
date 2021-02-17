import numpy as np
import socket

from select import select
from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import AckMessage, DataMessage, FrameHeaderMessage, FrameMessage, SimpleMessage
from ..base import RGBDFrameReceiver, SocketUtil
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionService:
    """A skeleton detection service to which detection requests can be made over a network."""

    # CONSTRUCTOR

    def __init__(self, frame_processor: Callable[[np.ndarray, np.ndarray, np.ndarray], List[Skeleton]],
                 port: int = 7852, *, debug: bool = False,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a skeleton detection service.

        :param frame_processor:     The function to use to detect skeletons in frames.
        :param port:                The port on which the service should listen for connections.
        :param debug:               Whether to print out debug messages.
        :param frame_decompressor:  An optional function to use to decompress received frames.
        """
        self.__debug: bool = debug
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__frame_processor: Callable[[np.ndarray, np.ndarray], List[Skeleton]] = frame_processor
        self.__port: int = port

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the service."""
        # Set up the server socket and listen for a connection.
        server_sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for a connection on 127.0.0.1:{self.__port}...")

        client_sock: Optional[socket.SocketType] = None

        while client_sock is None:
            timeout: float = 0.1
            readable, _, _ = select([server_sock], [], [], timeout)

            for s in readable:
                if s is server_sock:
                    client_sock, client_endpoint = server_sock.accept()
                    print(f"Accepted connection from client @ {client_endpoint}")

        # Once a client has connected, process any detection requests received from it.
        connection_ok: bool = True
        receiver: RGBDFrameReceiver = RGBDFrameReceiver()
        skeletons: Optional[List[Skeleton]] = None

        while connection_ok:
            # First, try to read a control message from the client.
            control_msg: SkeletonControlMessage = SkeletonControlMessage()
            connection_ok = SocketUtil.read_message(client_sock, control_msg)

            # If that succeeds:
            if connection_ok:
                # Extract the control message value to tell us whether this is the start or end of a detection.
                value: int = control_msg.extract_value()

                # If this is the start of a detection:
                if value == SkeletonControlMessage.BEGIN_DETECTION:
                    if self.__debug:
                        print("Begin Detection")

                    # Try to read a frame header message.
                    max_images: int = 2
                    header_msg: FrameHeaderMessage = FrameHeaderMessage(max_images)
                    connection_ok = SocketUtil.read_message(client_sock, header_msg)

                    # If that succeeds:
                    if connection_ok:
                        # Set up a frame message accordingly.
                        image_shapes: List[Tuple[int, int, int]] = header_msg.get_image_shapes()
                        image_byte_sizes: List[int] = header_msg.get_image_byte_sizes()
                        frame_msg: FrameMessage = FrameMessage(image_shapes, image_byte_sizes)

                        # Try to read the contents of the frame message from the client.
                        connection_ok = SocketUtil.read_message(client_sock, frame_msg)

                        # If that succeeds:
                        if connection_ok:
                            # Send an acknowledgement to the client.
                            connection_ok = SocketUtil.write_message(client_sock, AckMessage())

                            # Decompress the frame as necessary.
                            decompressed_frame_msg: FrameMessage = frame_msg
                            if self.__frame_decompressor is not None:
                                decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                            # Detect any people who are present in the frame.
                            receiver(decompressed_frame_msg)

                            skeletons = self.__frame_processor(
                                receiver.get_rgb_image(), receiver.get_depth_image(), receiver.get_pose()
                            )

                # Otherwise, if this is the end of a detection:
                else:
                    if self.__debug:
                        print(f"End Detection")

                    # Assuming the skeletons have previously been detected (if not, it's because the client has
                    # erroneously called end_detection prior to begin_detection):
                    if skeletons is not None:
                        # Send them across to the client.
                        data: np.ndarray = np.frombuffer(bytes(repr(skeletons), "utf-8"), dtype=np.uint8)
                        data_msg: DataMessage = DataMessage(len(data))
                        np.copyto(data_msg.get_data(), data)

                        connection_ok = \
                            SocketUtil.write_message(client_sock, SimpleMessage[int](len(data))) and \
                            SocketUtil.write_message(client_sock, data_msg)

                        # Now that we've sent the skeletons, clear them so that they don't get sent to the client
                        # again erroneously in future frames.
                        skeletons = None
