import cv2
import numpy as np

from typing import List

from smg.comms.skeletons import SkeletonDetectionService
from smg.skeletons import Skeleton


def frame_processor(colour_image: np.ndarray, depth_image: np.ndarray, world_from_camera: np.ndarray) -> List[Skeleton]:
    cv2.imshow("Colour Image", colour_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return []


def main() -> None:
    with SkeletonDetectionService(frame_processor=frame_processor) as service:
        service.start()
        while not service.should_terminate():
            pass


if __name__ == "__main__":
    main()
