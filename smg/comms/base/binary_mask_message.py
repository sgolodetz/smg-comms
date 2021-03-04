import numpy as np

from typing import Tuple

from .message import Message


class BinaryMaskMessage(Message):
    """A message containing a binary mask image."""

    # CONSTRUCTOR

    def __init__(self, mask_shape: Tuple[int, int]):
        """
        TODO

        :param mask_shape:  TODO
        """
        super().__init__()

        self.__mask_shape: Tuple[int, int] = mask_shape

        self._data = np.zeros(int(np.ceil(np.product(mask_shape) / 8)), dtype=np.uint8)

    # PUBLIC METHODS

    def get_mask(self) -> np.ndarray:
        unpacked: np.ndarray = (np.unpackbits(self._data, count=np.product(self.__mask_shape)) * 255).astype(np.uint8)
        return unpacked.reshape(self.__mask_shape)

    def set_mask(self, mask: np.ndarray) -> None:
        if mask.shape == self.__mask_shape:
            np.copyto(self._data, np.packbits(mask))
        else:
            raise RuntimeError(f"The binary mask has shape {mask.shape} instead of {self.__mask_shape}")
