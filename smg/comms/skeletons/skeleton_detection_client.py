import numpy as np
import socket

from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import FrameMessage, SocketUtil
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
            self.__sock.settimeout(timeout)
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

    def begin_detection(self, image: np.ndarray) -> str:
        connection_ok: bool = True
        connection_ok = connection_ok and SocketUtil.write_message(
            self.__sock, SkeletonControlMessage.begin_detection()
        )
        return ""

    def end_detection(self, request_token: str) -> Optional[List[Skeleton]]:
        pass

    def terminate(self) -> None:
        """Tell the client to terminate."""
        if self.__alive:
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False
