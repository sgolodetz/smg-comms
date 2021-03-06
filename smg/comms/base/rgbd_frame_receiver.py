import numpy as np

from typing import Optional

from smg.utility import ImageUtil

from .frame_message import FrameMessage
from .rgbd_frame_message_util import RGBDFrameMessageUtil


class RGBDFrameReceiver:
    """A receiver of RGB-D frames, used to extract them from frame messages and store them for later use."""

    # CONSTRUCTOR

    def __init__(self):
        """Construct an RGB-D frame receiver."""
        self.__frame_idx = None        # type: Optional[int]
        self.__frame_timestamp = None  # type: Optional[float]
        self.__rgb_image = None        # type: Optional[np.ndarray]
        self.__depth_image = None      # type: Optional[np.ndarray]
        self.__pose = None             # type: Optional[np.ndarray]

    # SPECIAL METHODS

    def __call__(self, msg: FrameMessage) -> None:
        """
        Apply the receiver to a frame message.

        .. note::
            This extracts an RGB-D frame from the frame message and stores it in the receiver.

        :param msg: The frame message.
        """
        self.__frame_idx, self.__frame_timestamp, self.__rgb_image, short_depth_image, self.__pose = \
            RGBDFrameMessageUtil.extract_frame_data(msg)
        self.__depth_image = ImageUtil.from_short_depth(short_depth_image)

    # PUBLIC METHODS

    def get_depth_image(self) -> np.ndarray:
        """
        Get the depth image of the RGB-D frame.

        :return:    The depth image of the RGB-D frame.
        """
        return self.__depth_image

    def get_frame_index(self) -> int:
        """
        Get the index of the RGB-D frame.

        :return:    The index of the RGB-D frame.
        """
        return self.__frame_idx

    def get_frame_timestamp(self) -> Optional[float]:
        """
        Get the timestamp of the RGB-D frame.

        .. note::
            If a timestamp wasn't provided when the frame was sent, this will return None.

        :return:    The timestamp of the RGB-D frame.
        """
        return self.__frame_timestamp

    def get_pose(self) -> np.ndarray:
        """
        Get the pose of the RGB-D frame.

        :return:    The pose of the RGB-D frame.
        """
        return self.__pose

    def get_rgb_image(self) -> np.ndarray:
        """
        Get the colour image of the RGB-D frame.

        :return:    The colour image of the RGB-D frame.
        """
        return self.__rgb_image
