# -*- coding: future_annotations -*-

from ..base import SimpleMessage


class SkeletonControlMessage(SimpleMessage[int]):
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, value: int):
        """
        TODO

        :param value:   TODO
        """
        super().__init__(value, int)

    # PUBLIC STATIC METHODS

    # noinspection PyUnresolvedReferences
    @staticmethod
    def begin_detection() -> SkeletonControlMessage:
        return SkeletonControlMessage(0)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def end_detection() -> SkeletonControlMessage:
        return SkeletonControlMessage(1)
