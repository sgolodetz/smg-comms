import cv2
import numpy as np

from typing import List

from smg.comms.skeletons import SkeletonDetectionClient
from smg.skeletons import Skeleton


def main() -> None:
    with SkeletonDetectionClient() as client:
        frame_idx: int = 0
        image: np.ndarray = cv2.imread("C:/smglib/smg-lcrnet/smg/external/lcrnet/058017637.jpg")
        world_from_camera: np.ndarray = np.eye(4)
        while True:
            if client.begin_detection(frame_idx, image, world_from_camera):
                skeletons: List[Skeleton] = client.end_detection(frame_idx)
                print(f"{frame_idx}: {skeletons}")
            frame_idx += 1


if __name__ == "__main__":
    main()
