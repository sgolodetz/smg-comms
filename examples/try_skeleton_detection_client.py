import numpy as np

from smg.comms.skeletons import SkeletonDetectionClient


def main() -> None:
    with SkeletonDetectionClient() as client:
        frame_idx: int = 0
        image: np.ndarray = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)
        world_from_camera: np.ndarray = np.eye(4)
        if client.begin_detection(frame_idx, image, world_from_camera):
            print(client.end_detection(frame_idx))


if __name__ == "__main__":
    main()
