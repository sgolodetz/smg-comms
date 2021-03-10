import numpy as np
import socket

from typing import Callable, List, Optional, Tuple

from smg.skeletons import Skeleton

from ..base import *
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
        self.__people_mask_shape: Optional[Tuple[int, int]] = None

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

    def begin_detection(self, colour_image: np.ndarray, world_from_camera: np.ndarray) -> bool:
        """
        Try to request that the remote skeleton detection service detect any skeletons in the specified colour image.

        .. note::
            This will return False iff the connection drops before the request can be completed.

        :param colour_image:        The colour image.
        :param world_from_camera:   The pose from which the image was captured.
        :return:                    True, if the detection was successfully requested, or False otherwise.
        """
        # Make the frame message.
        dummy_frame_idx: int = -1
        dummy_depth_image: np.ndarray = np.zeros(colour_image.shape[:2], dtype=np.uint16)
        frame_msg: FrameMessage = RGBDFrameMessageUtil.make_frame_message(
            dummy_frame_idx, colour_image, dummy_depth_image, world_from_camera
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

        # If that succeeded, store the expected people mask shape for later.
        if connection_ok:
            self.__people_mask_shape = colour_image.shape[:2]

        return connection_ok

    def detect_skeletons(self, colour_image: np.ndarray, world_from_camera: np.ndarray) \
            -> Optional[Tuple[List[Skeleton], np.ndarray]]:
        """
        Try to use the remote skeleton detection service to detect any skeletons in the specified colour image.

        :param colour_image:        The colour image.
        :param world_from_camera:   The pose from which the image was captured.
        :return:                    A tuple consisting of a list of skeletons and a people mask, if the detection
                                    succeeded, or None otherwise.
        """
        if self.begin_detection(colour_image, world_from_camera):
            return self.end_detection()
        else:
            return None

    def end_detection(self) -> Optional[Tuple[List[Skeleton], np.ndarray]]:
        """
        Try to request that the remote skeleton detection service send across any skeletons that it has just detected.

        :return:    A tuple consisting of a list of skeletons and a people mask, if successful, or None otherwise.
        """
        # Make a local copy of the expected people mask shape, if any, and reset the global one.
        people_mask_shape: Optional[Tuple[int, int]] = self.__people_mask_shape
        self.__people_mask_shape = None

        # If there isn't an expected people mask shape, there wasn't a previous successful call to
        # begin_detection, so early out.
        if people_mask_shape is None:
            return None

        # First send the end detection message, then read the size of the skeleton data that the service
        # wants to send across.
        data_size_msg: SimpleMessage[int] = SimpleMessage[int]()
        connection_ok: bool = \
            SocketUtil.write_message(self.__sock, SkeletonControlMessage.end_detection()) and \
            SocketUtil.read_message(self.__sock, data_size_msg)

        # If that succeeds:
        if connection_ok:
            # Read the skeleton data itself, as well as the people mask.
            data_msg: DataMessage = DataMessage(data_size_msg.extract_value())
            mask_msg: BinaryMaskMessage = BinaryMaskMessage(people_mask_shape)
            connection_ok = \
                SocketUtil.read_message(self.__sock, data_msg) and \
                SocketUtil.read_message(self.__sock, mask_msg)

            # If that succeeds:
            if connection_ok:
                # Construct a list of skeletons from the data.
                data: str = str(data_msg.get_data().tobytes(), "utf-8")
                skeletons: List[Skeleton] = eval(
                    data, {'array': np.array, 'Keypoint': Skeleton.Keypoint, 'Skeleton': Skeleton}
                )

                # Extract the people mask.
                people_mask: np.ndarray = mask_msg.get_mask()

                # TEMPORARY: Show the people mask.
                import cv2
                cv2.imshow("People Mask", people_mask)
                cv2.waitKey(1)

                return skeletons, people_mask

        # If anything goes wrong, return None.
        return None

    def set_calibration(self, image_size: Tuple[int, int], intrinsics: Tuple[float, float, float, float]) -> bool:
        """
        Try to send the camera calibration to the remote skeleton detection service.

        :param image_size:  The image size.
        :param intrinsics:  The camera intrinsics, as an (fx, fy, cx, cy) tuple.
        :return:            True, if the camera calibration was successfully sent, or False otherwise.
        """
        calib_msg: CalibrationMessage = RGBDFrameMessageUtil.make_calibration_message(
            image_size, image_size, intrinsics, intrinsics
        )

        ack_msg: AckMessage = AckMessage()

        connection_ok: bool = \
            SocketUtil.write_message(self.__sock, SkeletonControlMessage.set_calibration()) and \
            SocketUtil.write_message(self.__sock, calib_msg) and \
            SocketUtil.read_message(self.__sock, ack_msg)

        return connection_ok

    def terminate(self) -> None:
        """Tell the detector to terminate."""
        if self.__alive:
            self.__sock.shutdown(socket.SHUT_RDWR)
            self.__sock.close()
            self.__alive = False
