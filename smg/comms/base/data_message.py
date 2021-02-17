import numpy as np

from .message import Message


class DataMessage(Message):
    """A message containing a variable amount of data of an unspecified type."""

    # CONSTRUCTOR

    def __init__(self, size: int):
        """
        Construct a data message.

        :param size:    The size of the message (in bytes).
        """
        super().__init__()
        self._data = np.zeros(size, dtype=np.uint8)
