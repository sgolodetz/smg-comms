import numpy as np

from typing import Optional

from smg.comms.skeletons import SkeletonDetectionClient


def main() -> None:
    with SkeletonDetectionClient() as client:
        token: Optional[int] = client.begin_detection(np.zeros((480, 640, 3), dtype=np.uint8))
        if token is not None:
            client.end_detection(token)


if __name__ == "__main__":
    main()
