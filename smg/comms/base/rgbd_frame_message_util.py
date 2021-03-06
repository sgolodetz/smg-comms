import cv2
import numpy as np
import struct

from typing import List, Optional, Tuple

from .calibration_message import CalibrationMessage
from .frame_message import FrameMessage


class RGBDFrameMessageUtil:
    """Utility functions related to RGB-D frame messages."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compress_frame_message(msg: FrameMessage) -> FrameMessage:
        """
        Compress an uncompressed RGB-D frame message.

        :param msg: The message to compress.
        :return:    The compressed message.
        """
        # Extract the relevant data from the uncompressed frame message.
        frame_idx, frame_timestamp, rgb_image, depth_image, pose = RGBDFrameMessageUtil.extract_frame_data(msg)

        # Compress the RGB and depth images.
        compressed_rgb_image = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]  # type: np.ndarray
        compressed_depth_image = cv2.imencode(".png", depth_image)[1]                              # type: np.ndarray

        # Construct and return the compressed message.
        compressed_msg = FrameMessage(
            msg.get_image_shapes(), [len(compressed_rgb_image), len(compressed_depth_image)]
        )
        compressed_msg.set_frame_index(frame_idx)
        compressed_msg.set_frame_timestamp(frame_timestamp)
        compressed_msg.set_image_data(0, compressed_rgb_image.flatten())
        compressed_msg.set_pose(0, pose)
        compressed_msg.set_image_data(1, compressed_depth_image.flatten())
        compressed_msg.set_pose(1, pose)

        return compressed_msg

    @staticmethod
    def decompress_frame_message(msg: FrameMessage) -> FrameMessage:
        """
        Decompress a compressed RGB-D frame message.

        :param msg: The message to decompress.
        :return:    The decompressed message.
        """
        # Extract the relevant data from the compressed frame message.
        frame_idx = msg.get_frame_index()               # type: int
        frame_timestamp = msg.get_frame_timestamp()     # type: Optional[float]
        compressed_rgb_image = msg.get_image_data(0)    # type: np.ndarray
        compressed_depth_image = msg.get_image_data(1)  # type: np.ndarray
        pose = msg.get_pose(0)                          # type: np.ndarray

        # Uncompress the RGB and depth images.
        rgb_image = cv2.imdecode(compressed_rgb_image, cv2.IMREAD_COLOR)                           # type: np.ndarray
        depth_image = cv2.imdecode(compressed_depth_image, cv2.IMREAD_ANYDEPTH).astype(np.uint16)  # type: np.ndarray

        # Construct and return the decompressed message.
        decompressed_msg = FrameMessage(
            msg.get_image_shapes(),
            [rgb_image.nbytes, depth_image.nbytes]
        )
        RGBDFrameMessageUtil.fill_frame_message(
            frame_idx, rgb_image, depth_image, pose, decompressed_msg, frame_timestamp=frame_timestamp
        )

        return decompressed_msg

    @staticmethod
    def extract_frame_data(msg: FrameMessage) -> Tuple[int, Optional[float], np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the relevant data from an uncompressed RGB-D frame message.

        :param msg: The uncompressed RGB-D frame message.
        :return:    A tuple consisting of the frame index, the frame timestamp, the RGB image, the depth image
                    and the pose.
        """
        frame_idx = msg.get_frame_index()  # type: int
        frame_timestamp = msg.get_frame_timestamp()  # type: Optional[float]
        rgb_image = msg.get_image_data(0).reshape(msg.get_image_shapes()[0])  # type: np.ndarray
        depth_image = msg.get_image_data(1).view(np.uint16).reshape(msg.get_image_shapes()[1][:2])  # type: np.ndarray
        pose = msg.get_pose(0)  # type: np.ndarray

        # Note: It's extremely important that we return *copies* of the data here, since the versions in
        #       the message may change once this method returns. (The context is that the message comes
        #       from a pool, and it will be returned to the pool once the lock that's held whilst this
        #       method is called is released. See also MappingClientHandler.get_frame.)
        return frame_idx, frame_timestamp, rgb_image.copy(), depth_image.copy(), pose.copy()

    @staticmethod
    def fill_frame_message(frame_idx: int, rgb_image: np.ndarray, depth_image: np.ndarray, pose: np.ndarray,
                           msg: FrameMessage, *, frame_timestamp: Optional[float] = None) -> None:
        """
        Fill an uncompressed RGB-D frame message with the necessary data.

        :param frame_idx:       The frame index.
        :param rgb_image:       The RGB-D image.
        :param depth_image:     The depth image (with dtype np.uint16).
        :param pose:            The pose.
        :param msg:             The uncompressed RGB-D frame message.
        :param frame_timestamp: The frame timestamp (this is optional, and can be None if not known).
        """
        msg.set_frame_index(frame_idx)
        msg.set_frame_timestamp(frame_timestamp)
        msg.set_image_data(0, rgb_image.reshape(-1))
        msg.set_pose(0, pose)
        msg.set_image_data(1, depth_image.reshape(-1).view(np.uint8))
        msg.set_pose(1, pose)

    @staticmethod
    def make_calibration_message(rgb_image_size: Tuple[int, int], depth_image_size: Tuple[int, int],
                                 rgb_intrinsics: Tuple[float, float, float, float],
                                 depth_intrinsics: Tuple[float, float, float, float]) -> CalibrationMessage:
        """
        Make a calibration message that specifies the shapes, intrinsics and element byte sizes
        for an RGB-D image pair.

        :param rgb_image_size:      The size of the RGB images, as a (width, height) tuple.
        :param depth_image_size:    The size of the depth images, as a (width, height) tuple.
        :param rgb_intrinsics:      The RGB camera intrinsics.
        :param depth_intrinsics:    The depth camera intrinsics.
        :return:                    The calibration message.
        """
        calib_msg = CalibrationMessage()

        # noinspection PyTypeChecker
        calib_msg.set_image_shapes([rgb_image_size[::-1] + (3,), depth_image_size[::-1] + (1,)])
        calib_msg.set_intrinsics([rgb_intrinsics, depth_intrinsics])
        calib_msg.set_element_byte_sizes([struct.calcsize("<B"), struct.calcsize("<H")])

        return calib_msg

    @staticmethod
    def make_frame_message(frame_idx: int, rgb_image: np.ndarray, depth_image: np.ndarray,
                           pose: np.ndarray) -> FrameMessage:
        """
        Make an uncompressed RGB-D frame message and fill it with the specified data.

        :param frame_idx:   The frame index.
        :param rgb_image:   The RGB-D image.
        :param depth_image: The depth image (with dtype np.uint16).
        :param pose:        The pose.
        :return:            The uncompressed RGB-D frame message.
        """
        image_shapes = [rgb_image.shape, depth_image.shape + (1,)]  # type: List[Tuple[int, int, int]]
        element_byte_sizes = [struct.calcsize("<B"), struct.calcsize("<H")]  # type: List[int]
        image_byte_sizes = [np.prod(s) * b for s, b in zip(image_shapes, element_byte_sizes)]  # type: List[int]
        msg = FrameMessage(image_shapes, image_byte_sizes)
        RGBDFrameMessageUtil.fill_frame_message(frame_idx, rgb_image, depth_image, pose, msg)
        return msg
