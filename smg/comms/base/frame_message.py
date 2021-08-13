import numpy as np
import struct

from typing import List, Optional, Tuple

from .message import Message


class FrameMessage(Message):
    """A message containing a single frame of data (frame index + poses + images)."""

    # CONSTRUCTOR

    def __init__(self, image_shapes: List[Tuple[int, int, int]], image_byte_sizes: List[int]):
        """
        Construct a frame message.

        .. note::
            A frame message can in principle contain any number of images, up to the maximum specified
            in the calibration message. If the input lists passed in to the constructor are shorter
            than the maximum, they will be padded for storage in the frame message.
        .. note::
            The image byte sizes refer to the actual storage requirements for the images in the message.
            If compressed images are to be stored, the byte sizes passed in will be the compressed ones.

        :param image_shapes:        The shapes of the images that will be stored in the frame message.
        :param image_byte_sizes:    The overall byte sizes of the images that will be stored in the frame message.
        """
        super().__init__()

        self.__image_shapes = image_shapes  # type: List[Tuple[int, int, int]]
        self.__image_byte_sizes = image_byte_sizes  # type: List[int]

        # The frame index segment consists of a single integer denoting the frame index.
        self.__frame_index_fmt = "<i"  # type: str

        # The frame timestamp segment consists of a single double denoting the frame timestamp.
        self.__frame_timestamp_fmt = "<d"  # type: str

        # The poses segment consists of a list of poses, each denoting a 4x4 matrix.
        self.__pose_byte_size = struct.calcsize("<ffffffffffffffff")  # type: int
        self.__poses_fmt = "<" + "ffffffffffffffff" * len(image_shapes)  # type: str

        self.__frame_index_segment = (
            0, struct.calcsize(self.__frame_index_fmt)
        )  # type: Tuple[int, int]
        self.__frame_timestamp_segment = (
            Message._end_of(self.__frame_index_segment), struct.calcsize(self.__frame_timestamp_fmt)
        )  # type: Tuple[int, int]
        self.__poses_segment = (
            Message._end_of(self.__frame_timestamp_segment), struct.calcsize(self.__poses_fmt)
        )  # type: Tuple[int, int]
        self.__images_segment = (
            Message._end_of(self.__poses_segment), sum(self.__image_byte_sizes)
        )  # type: Tuple[int, int]

        self._data = np.zeros(Message._end_of(self.__images_segment), dtype=np.uint8)

    # PUBLIC METHODS

    def get_frame_index(self) -> int:
        """
        Get the frame index from the message.

        :return:    The frame index.
        """
        return struct.unpack_from(self.__frame_index_fmt, self._data, self.__frame_index_segment[0])[0]

    def get_frame_timestamp(self) -> Optional[float]:
        """
        Get the frame timestamp from the message.

        :return:    The frame timestamp.
        """
        frame_timestamp: float = struct.unpack_from(
            self.__frame_timestamp_fmt, self._data, self.__frame_timestamp_segment[0]
        )[0]
        return frame_timestamp if frame_timestamp >= 0 else None

    def get_image_byte_sizes(self) -> List[int]:
        """
        Get the byte sizes of the images (as stored in the message).

        :return:    The byte sizes of the images (as stored in the message).
        """
        return self.__image_byte_sizes

    def get_image_data(self, image_idx: int) -> np.ndarray:
        """
        Get the data for the specified image.

        :param image_idx:   The index of the image whose data we want to look up.
        :return:            The data for the specified image.
        """
        start = self.__images_segment[0]  # type: int
        for i in range(image_idx):
            start += self.__image_byte_sizes[i]
        end = start + self.__image_byte_sizes[image_idx]  # type: int
        return self._data[start:end]

    def get_image_shapes(self) -> List[Tuple[int, int, int]]:
        """
        Get the image shapes.

        :return:    The image shapes.
        """
        return self.__image_shapes

    def get_pose(self, image_idx: int) -> np.ndarray:
        """
        Get the pose of the specified image.

        :param image_idx:   The index of the image whose pose want to look up.
        :return:            The pose of the specified image, as a 4x4 matrix.
        """
        return self.__get_pose_data(image_idx).view(np.float32).reshape((4, 4))

    def set_frame_index(self, frame_index: int) -> None:
        """
        Copy a frame index into the appropriate byte segment in the message.

        :param frame_index: The frame index.
        """
        struct.pack_into(self.__frame_index_fmt, self._data, self.__frame_index_segment[0], frame_index)

    def set_frame_timestamp(self, frame_timestamp: Optional[float]) -> None:
        """
        Copy a frame timestamp into the appropriate byte segment in the message.

        :param frame_timestamp: The frame timestamp.
        """
        struct.pack_into(
            self.__frame_timestamp_fmt, self._data, self.__frame_timestamp_segment[0],
            frame_timestamp if frame_timestamp is not None else -1.0
        )

    def set_image_data(self, image_idx: int, image_data: np.ndarray) -> None:
        """
        Copy the data for the specified image into the appropriate byte segment in the message.

        :param image_idx:   The index of the image whose data we want to set.
        :param image_data:  The data for the specified image.
        """
        np.copyto(self.get_image_data(image_idx), image_data)

    def set_pose(self, image_idx: int, pose: np.ndarray) -> None:
        """
        Copy the pose for the specified image into the appropriate byte segment in the message.

        :param image_idx:   The index of the image whose pose we want to set.
        :param pose:        The pose for the specified image.
        """
        np.copyto(self.__get_pose_data(image_idx), pose.astype(np.float32).reshape(-1).view(np.uint8))

    # PRIVATE METHODS

    def __get_pose_data(self, image_idx: int) -> np.ndarray:
        """
        Get the pose data for the specified image.

        :param image_idx:   The index of the image whose pose data we want to look up.
        :return:            The pose data for the specified image, as a byte segment.
        """
        start = self.__poses_segment[0] + image_idx * self.__pose_byte_size  # type: int
        return self._data[start:start+self.__pose_byte_size]
