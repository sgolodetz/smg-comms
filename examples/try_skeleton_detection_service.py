from smg.comms.skeletons import SkeletonDetectionService


def main() -> None:
    with SkeletonDetectionService() as service:
        service.start()
        while True:
            pass


if __name__ == "__main__":
    main()
