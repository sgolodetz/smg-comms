import numpy as np
import socket

from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import AckMessage, DataMessage, FrameHeaderMessage, FrameMessage, SimpleMessage
from ..base import RGBDFrameMessageUtil, SocketUtil
from .skeleton_control_message import SkeletonControlMessage


class RemoteSkeletonDetector:
    """A skeleton detector that makes use of a remote skeleton detection service to operate."""

    # CONSTRUCTOR

    def __init__(self, endpoint: Tuple[str, int] = ("127.0.0.1", 7852), *, timeout: Optional[float] = None,
                 frame_compressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a remote skeleton detector.

        :param endpoint:            The service host and port, e.g. ("127.0.0.1", 7852).
        :param timeout:             An optional socket timeout (in seconds).
        :param frame_compressor:    An optional function to use to compress frames prior to transmission.
        """
        self.__alive: bool = False
        self.__frame_compressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_compressor

        try:
            # Try to connect to the service.
            self.__sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__sock.connect(endpoint)
            if timeout is not None:
                self.__sock.settimeout(timeout)
            self.__alive = True
        except ConnectionRefusedError:
            # If we couldn't connect to the service, raise an exception.
            raise RuntimeError("Error: Could not connect to the service")

    # DESTRUCTOR

    def __del__(self):
        """Destroy the detector."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the detector's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the detector at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def begin_detection(self, image: np.ndarray, world_from_camera: np.ndarray) -> bool:
        """
        Try to request that the remote skeleton detection service detect any skeletons in the specified image.

        .. note::
            This will return False iff the connection drops before the request can be completed.

        :param image:               The image.
        :param world_from_camera:   The pose from which the image was captured.
        :return:                    True, if the detection was successfully requested, or False otherwise.
        """
        # Make the frame message.
        dummy_frame_idx: int = -1
        frame_msg: FrameMessage = RGBDFrameMessageUtil.make_frame_message(
            dummy_frame_idx, image, np.zeros(image.shape[:2], dtype=np.uint16), world_from_camera
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

    def detect_skeletons(self, image: np.ndarray, world_from_camera: np.ndarray) -> Optional[List[Skeleton]]:
        """
        Try to use the remote skeleton detection service to detect any skeletons in the specified image.

        :param image:               The image.
        :param world_from_camera:   The pose from which the image was captured.
        :return:                    A list of skeletons, if the detection succeeded, or None otherwise.
        """
        if self.begin_detection(image, world_from_camera):
            return self.end_detection()
        else:
            return None

    def end_detection(self) -> Optional[List[Skeleton]]:
        """
        Try to request that the remote skeleton detection service send across any skeletons that it has just detected.

        :return:    The skeletons, if successful, or None otherwise.
        """
        # First send the end detection message, then read the size of the data that the service wants to send across.
        data_size_msg: SimpleMessage[int] = SimpleMessage[int]()
        connection_ok: bool = \
            SocketUtil.write_message(self.__sock, SkeletonControlMessage.end_detection()) and \
            SocketUtil.read_message(self.__sock, data_size_msg)

        # If that succeeds:
        if connection_ok:
            # Read the data itself.
            data_msg: DataMessage = DataMessage(data_size_msg.extract_value())
            connection_ok = SocketUtil.read_message(self.__sock, data_msg)

            # If that succeeds:
            if connection_ok:
                # Construct a list of skeletons from the data, and return it.
                data: str = str(data_msg.get_data().tobytes(), "utf-8")
                skeletons: List[Skeleton] = eval(
                    data, {'array': np.array, 'Keypoint': Skeleton.Keypoint, 'Skeleton': Skeleton}
                )
                return skeletons

        # If anything goes wrong, return None.
        return None

    def terminate(self) -> None:
        """Tell the client to terminate."""
        if self.__alive:
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False
