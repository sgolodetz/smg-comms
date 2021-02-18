# -*- coding: future_annotations -*-

from typing import Optional

from ..base import SimpleMessage


class SkeletonControlMessage(SimpleMessage[int]):
    """A message controlling the type of interaction a client wants to have with a skeleton detection service."""

    # CONSTANTS

    BEGIN_DETECTION: int = 0
    END_DETECTION: int = 1

    # CONSTRUCTOR

    def __init__(self, value: Optional[int] = None):
        """
        Construct a skeleton control message.

        :param value:   An integer value specifying the type of interaction a client wants to have with a skeleton
                        detection service.
        """
        super().__init__(value, int)

    # PUBLIC STATIC METHODS

    # noinspection PyUnresolvedReferences
    @staticmethod
    def begin_detection() -> SkeletonControlMessage:
        """
        Make a 'begin detection' control message.

        :return:    The control message.
        """
        return SkeletonControlMessage(SkeletonControlMessage.BEGIN_DETECTION)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def end_detection() -> SkeletonControlMessage:
        """
        Make an 'end detection' control message.

        :return:    The control message.
        """
        return SkeletonControlMessage(SkeletonControlMessage.END_DETECTION)
