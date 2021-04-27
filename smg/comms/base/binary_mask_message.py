import numpy as np

from typing import Tuple

from .message import Message


class BinaryMaskMessage(Message):
    """A message containing a binary mask."""

    # CONSTRUCTOR

    def __init__(self, mask_shape: Tuple[int, int]):
        """
        Construct a binary mask message.

        :param mask_shape:  The shape of the binary mask, namely a (height, width) tuple.
        """
        super().__init__()

        self.__mask_shape = mask_shape  # type: Tuple[int, int]

        # Allocate a buffer of the size needed to store the bit-packed mask.
        self._data = np.zeros(int(np.ceil(np.product(mask_shape) / 8)), dtype=np.uint8)

    # PUBLIC METHODS

    def get_mask(self) -> np.ndarray:
        """
        Extract the binary mask from the message.

        :return:    The binary mask.
        """
        unpacked = (
            np.unpackbits(self._data, count=np.product(self.__mask_shape)) * 255
        ).astype(np.uint8)  # type: np.ndarray
        return unpacked.reshape(self.__mask_shape)

    def set_mask(self, mask: np.ndarray) -> None:
        """
        Replace the binary mask in the message.

        .. note::
            The new mask must have the shape specified when the message was constructed.

        :param mask:            The binary mask with which to replace the one in the message.
        :raises RuntimeError:   If the new mask has the wrong shape.
        """
        if mask.shape == self.__mask_shape:
            np.copyto(self._data, np.packbits(mask))
        else:
            raise RuntimeError("The binary mask has shape {} instead of {}".format(mask.shape, self.__mask_shape))
