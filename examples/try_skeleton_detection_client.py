import cv2
import numpy as np

from smg.comms.skeletons import SkeletonDetectionClient


def main() -> None:
    with SkeletonDetectionClient() as client:
        frame_idx: int = 0
        image: np.ndarray = cv2.imread("C:/smglib/smg-lcrnet/smg/external/lcrnet/058017637.jpg")
        world_from_camera: np.ndarray = np.eye(4)
        if client.begin_detection(frame_idx, image, world_from_camera):
            print(client.end_detection(frame_idx))
        while True:
            pass


if __name__ == "__main__":
    main()
