from typing import Optional

from ..base import SimpleMessage


class SkeletonControlMessage(SimpleMessage[int]):
    """A message controlling the type of interaction a client wants to have with a skeleton detection service."""

    # CONSTANTS

    BEGIN_DETECTION = 0  # type: int
    END_DETECTION = 1    # type: int
    SET_CALIBRATION = 2  # type: int

    # CONSTRUCTOR

    def __init__(self, value: Optional[int] = None):
        """
        Construct a skeleton control message.

        :param value:   An integer value specifying the type of interaction a client wants to have with a skeleton
                        detection service.
        """
        super().__init__(int, value)

    # PUBLIC STATIC METHODS

    @staticmethod
    def begin_detection() -> "SkeletonControlMessage":
        """
        Make a 'begin detection' control message.

        :return:    The control message.
        """
        return SkeletonControlMessage(SkeletonControlMessage.BEGIN_DETECTION)

    @staticmethod
    def end_detection() -> "SkeletonControlMessage":
        """
        Make an 'end detection' control message.

        :return:    The control message.
        """
        return SkeletonControlMessage(SkeletonControlMessage.END_DETECTION)

    @staticmethod
    def set_calibration() -> "SkeletonControlMessage":
        """
        Make a 'set calibration' control message.

        :return:    The control message.
        """
        return SkeletonControlMessage(SkeletonControlMessage.SET_CALIBRATION)
