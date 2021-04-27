import numpy as np
import struct

from typing import Generic, Optional, TypeVar

from .message import Message


# TYPE VARIABLE

T = TypeVar('T')


# MAIN CLASS

class SimpleMessage(Message, Generic[T]):
    """A message containing a single value of a specified type."""

    # CONSTRUCTOR

    def __init__(self, t: type, value: Optional[T] = None):
        """
        Construct a simple message.

        .. note::
            It's not possible to infer the actual type of T here when this constructor is invoked by a
            derived constructor. Instead, the derived constructor must pass it in explicitly.

        :param t:       The actual type of T.
        :param value:   An optional initial message value.
        """
        super().__init__()

        # noinspection PyUnusedLocal
        size = 0  # type: int
        self.__fmt = ""  # type: str

        if issubclass(t, int):
            self.__fmt = "i"
            size = 4
        else:
            raise RuntimeError("Cannot construct SimpleMessage with unsupported type {}".format(t.__name__))

        self._data = np.zeros(size, dtype=np.uint8)

        if value is not None:
            self.set_value(value)

    # PUBLIC METHODS

    def extract_value(self) -> T:
        """
        Extract the message value.

        :return:    The message value.
        """
        return struct.unpack_from(self.__fmt, self._data, 0)[0]

    def set_value(self, value: T) -> None:
        """
        Set the message value.

        :param value:   The message value.
        """
        struct.pack_into(self.__fmt, self._data, 0, value)
