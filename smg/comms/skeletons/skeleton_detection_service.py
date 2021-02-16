import numpy as np
import socket

from select import select
from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import AckMessage, DataMessage, FrameHeaderMessage, FrameMessage, RGBDFrameMessageUtil, \
    SimpleMessage, SocketUtil
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionService:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, port: int = 7852, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None,
                 frame_processor: Callable[[np.ndarray, np.ndarray, np.ndarray], List[Skeleton]]):
        """
        TODO

        :param port:                TODO
        :param frame_decompressor:  TODO
        :param frame_processor:     TODO
        """
        self.__colour_image: Optional[np.ndarray] = None
        self.__depth_image: Optional[np.ndarray] = None
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__frame_processor: Callable[[np.ndarray, np.ndarray], List[Skeleton]] = frame_processor
        self.__port: int = port
        self.__skeletons: Optional[List[Skeleton]] = None
        self.__world_from_camera: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def start(self) -> None:
        """Start the service."""
        self.__run_service()

    # PRIVATE METHODS

    def __run_service(self) -> None:
        """Run the service thread."""
        # Set up the server socket and listen for connections.
        server_sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for connections on 127.0.0.1:{self.__port}...")

        client_sock: Optional[socket.SocketType] = None

        connection_ok: bool = True

        while connection_ok:
            if client_sock is not None:
                control_msg: SkeletonControlMessage = SkeletonControlMessage()
                connection_ok = SocketUtil.read_message(client_sock, control_msg)
                if connection_ok:
                    value: int = control_msg.extract_value()
                    if value == SkeletonControlMessage.BEGIN_DETECTION:
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
                                # Decompress the frame as necessary.
                                decompressed_frame_msg: FrameMessage = frame_msg
                                if self.__frame_decompressor is not None:
                                    decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                                # TODO: Comment here.
                                frame_idx, self.__colour_image, self.__depth_image, self.__world_from_camera = \
                                    RGBDFrameMessageUtil.extract_frame_data(decompressed_frame_msg)

                                # Send an acknowledgement to the client.
                                connection_ok = SocketUtil.write_message(client_sock, AckMessage())

                                self.__skeletons = self.__frame_processor(
                                    self.__colour_image, self.__depth_image, self.__world_from_camera
                                )
                    else:
                        frame_idx, blocking = np.abs(value) - 1, value >= 0
                        print(f"End Detection: Frame Index = {frame_idx}, blocking = {blocking}")

                        # TODO: If this isn't the first frame:
                        if self.__skeletons is not None:
                            data: np.ndarray = np.frombuffer(
                                bytes(repr(self.__skeletons), "utf-8"), dtype=np.uint8
                            )
                            data_msg: DataMessage = DataMessage(len(data))
                            np.copyto(data_msg.get_data(), data)

                            connection_ok = \
                                SocketUtil.write_message(client_sock, SimpleMessage[int](len(data))) and \
                                SocketUtil.write_message(client_sock, data_msg)
            else:
                timeout: float = 0.1
                readable, _, _ = select([server_sock], [], [], timeout)

                for s in readable:
                    if s is server_sock:
                        client_sock, client_endpoint = server_sock.accept()
                        print(f"Accepted connection from client @ {client_endpoint}")
