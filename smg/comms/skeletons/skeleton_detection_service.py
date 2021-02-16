import numpy as np
import socket
import threading

from select import select
from typing import Callable, List, Optional, Tuple

from ..base import AckMessage, FrameHeaderMessage, FrameMessage, RGBDFrameMessageUtil, SocketUtil
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionService:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, port: int = 7852, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        TODO

        :param port:                TODO
        :param frame_decompressor:  TODO
        """
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__port: int = port
        self.__service_thread: threading.Thread = threading.Thread(target=self.__run_service)
        self.__should_terminate: threading.Event = threading.Event()

        self.__lock: threading.Lock = threading.Lock()

    # DESTRUCTOR

    def __del__(self):
        """Destroy the service."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the service's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the service at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def start(self) -> None:
        """Start the service."""
        self.__service_thread.start()

    def terminate(self) -> None:
        """Tell the service to terminate."""
        with self.__lock:
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()
                self.__service_thread.join()

    # PRIVATE METHODS

    def __run_service(self) -> None:
        """Run the service."""
        # Set up the server socket and listen for connections.
        server_sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for connections on 127.0.0.1:{self.__port}...")

        client_sock: Optional[socket.SocketType] = None

        connection_ok: bool = True

        while connection_ok and not self.__should_terminate.is_set():
            if client_sock is not None:
                control_msg: SkeletonControlMessage = SkeletonControlMessage()
                connection_ok = SocketUtil.read_message(client_sock, control_msg)
                if connection_ok:
                    value: int = control_msg.extract_value()
                    if value == SkeletonControlMessage.BEGIN_DETECTION:
                        print("Begin Detection")
                        # TODO: Receive the actual image and store the detection request.

###
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
                                frame_idx, image, _, world_from_camera = RGBDFrameMessageUtil.extract_frame_data(
                                    decompressed_frame_msg
                                )

                                # TEMPORARY
                                import cv2
                                cv2.imshow("Image", image)
                                cv2.waitKey()

                                # Send an acknowledgement to the client.
                                connection_ok = SocketUtil.write_message(client_sock, AckMessage())
###
                    else:
                        frame_idx, blocking = np.abs(value) - 1, value >= 0
                        print(f"End Detection: Frame Index = {frame_idx}, blocking = {blocking}")
                        # TODO: Check whether the detection request has been processed yet.
            else:
                timeout: float = 0.1
                readable, _, _ = select([server_sock], [], [], timeout)
                if self.__should_terminate.is_set():
                    break

                for s in readable:
                    if s is server_sock:
                        client_sock, client_endpoint = server_sock.accept()
                        print(f"Accepted connection from client @ {client_endpoint}")
