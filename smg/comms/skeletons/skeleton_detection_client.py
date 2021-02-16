import numpy as np
import socket

from timeit import default_timer as timer
from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import AckMessage, DataMessage, FrameHeaderMessage, FrameMessage, RGBDFrameMessageUtil, \
    SimpleMessage, SocketUtil
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionClient:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7852), *, timeout: int = 10,
                 frame_compressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        self.__alive: bool = False
        self.__frame_compressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_compressor

        try:
            self.__sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.connect(endpoint)
            # self.__sock.settimeout(timeout)
            self.__alive = True
        except ConnectionRefusedError:
            raise RuntimeError("Error: Could not connect to the service")

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

    def begin_detection(self, frame_idx: int, image: np.ndarray, world_from_camera: np.ndarray) -> bool:
        # Make the frame message.
        frame_msg: FrameMessage = RGBDFrameMessageUtil.make_frame_message(
            frame_idx, image, np.zeros(image.shape[:2], dtype=np.uint16), world_from_camera
        )

        # If requested, compress the frame prior to transmission.
        compressed_frame_msg: FrameMessage = frame_msg
        if self.__frame_compressor is not None:
            compressed_frame_msg = self.__frame_compressor(frame_msg)

        # Make the frame header message.
        max_images: int = 2
        header_msg: FrameHeaderMessage = FrameHeaderMessage(max_images)
        header_msg.set_image_byte_sizes(compressed_frame_msg.get_image_byte_sizes())
        header_msg.set_image_shapes(compressed_frame_msg.get_image_shapes())

        # First send the begin detection message, then send the frame header message, then send the frame message,
        # then wait for an acknowledgement from the service. We chain all of these with 'and' so as to early out
        # in case of failure.
        connection_ok: bool = True
        ack_msg: AckMessage = AckMessage()

        connection_ok = connection_ok and \
            SocketUtil.write_message(self.__sock, SkeletonControlMessage.begin_detection()) and \
            SocketUtil.write_message(self.__sock, header_msg) and \
            SocketUtil.write_message(self.__sock, compressed_frame_msg) and \
            SocketUtil.read_message(self.__sock, ack_msg)

        return connection_ok

    def detect_skeletons(self, frame_idx: int, image: np.ndarray,
                         world_from_camera: np.ndarray) -> Optional[List[Skeleton]]:
        if self.begin_detection(frame_idx, image, world_from_camera):
            return self.end_detection(frame_idx)
        else:
            return None

    def end_detection(self, frame_idx: int, *, blocking: bool = True) -> Optional[List[Skeleton]]:
        data_size_msg: SimpleMessage[int] = SimpleMessage[int]()
        connection_ok: bool = \
            SocketUtil.write_message(
                self.__sock, SkeletonControlMessage.end_detection(frame_idx + 1, blocking=blocking)
            ) and \
            SocketUtil.read_message(self.__sock, data_size_msg)

        if connection_ok:
            data_msg: DataMessage = DataMessage(data_size_msg.extract_value())
            connection_ok = SocketUtil.read_message(self.__sock, data_msg)
            if connection_ok:
                start = timer()
                data: str = str(data_msg.get_data().tobytes(), "utf-8")
                skeletons: List[Skeleton] = eval(
                    data, {'array': np.array, 'Keypoint': Skeleton.Keypoint, 'Skeleton': Skeleton}
                )
                end = timer()
                print(f"Decode Time: {end - start}s")
                return skeletons

        return []

    def terminate(self) -> None:
        """Tell the client to terminate."""
        if self.__alive:
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False
