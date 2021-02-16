import cv2
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional

from smg.comms.skeletons import SkeletonDetectionClient
from smg.skeletons import Skeleton


def main() -> None:
    with SkeletonDetectionClient() as client:
        frame_idx: int = 0
        # image: np.ndarray = cv2.imread("C:/smglib/smg-lcrnet/smg/external/lcrnet/058017637.jpg")
        image: np.ndarray = cv2.imread("D:/LCRNet_v2.0/skeleton.png")
        world_from_camera: np.ndarray = np.eye(4)
        while True:
            start = timer()
            skeletons: Optional[List[Skeleton]] = client.detect_skeletons(frame_idx, image, world_from_camera)
            end = timer()

            if skeletons is not None:
                print(f"{frame_idx}: Remote Detection Time: {end - start}s")
                print(f"{frame_idx}: {skeletons}")

            frame_idx += 1


if __name__ == "__main__":
    main()
