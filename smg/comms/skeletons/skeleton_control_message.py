# -*- coding: future_annotations -*-

from typing import Optional

from ..base import SimpleMessage


class SkeletonControlMessage(SimpleMessage[int]):
    """TODO"""

    # ENUMERATION VALUES

    BEGIN_DETECTION: int = 0

    # CONSTRUCTOR

    def __init__(self, value: Optional[int] = None):
        """
        TODO

        :param value:   TODO
        """
        super().__init__(value, int)

    # PUBLIC STATIC METHODS

    # noinspection PyUnresolvedReferences
    @staticmethod
    def begin_detection() -> SkeletonControlMessage:
        return SkeletonControlMessage(SkeletonControlMessage.BEGIN_DETECTION)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def end_detection(frame_idx: int, *, blocking: bool) -> SkeletonControlMessage:
        return SkeletonControlMessage(frame_idx if blocking else -frame_idx)
