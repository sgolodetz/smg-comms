from smg.comms.skeletons import SkeletonDetectionClient


def main() -> None:
    with SkeletonDetectionClient() as client:
        pass


if __name__ == "__main__":
    main()
