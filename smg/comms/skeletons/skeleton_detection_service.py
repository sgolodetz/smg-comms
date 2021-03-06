import numpy as np
import socket

from OpenGL.GL import *
from select import select
from typing import Callable, List, Optional, Tuple

from smg.skeletons import PeopleMaskRenderer, Skeleton3D, SkeletonRenderer

from ..base import *
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionService:
    """A skeleton detection service to which a single client can make detection requests over a network."""

    # CONSTRUCTOR

    def __init__(self, frame_processor: Callable[
                    [int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]],
                    Tuple[List[Skeleton3D], Optional[np.ndarray]]
                 ],
                 port: int = 7852, *, debug: bool = False,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None,
                 post_client_hook: Optional[Callable[[], None]] = None):
        """
        Construct a skeleton detection service.

        :param frame_processor:     The function to use to detect skeletons in frames.
        :param port:                The port on which the service should listen for a connection.
        :param debug:               Whether to print out debug messages.
        :param frame_decompressor:  An optional function to use to decompress received frames.
        :param post_client_hook:    An optional function to call each time a client disconnects.
        """
        self.__debug = debug                                # type: bool
        self.__frame_decompressor = frame_decompressor      # type: Optional[Callable[[FrameMessage], FrameMessage]]
        self.__frame_processor = frame_processor            # type: Callable[[int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]], Tuple[List[Skeleton3D], Optional[np.ndarray]]]
        self.__people_mask_renderer = PeopleMaskRenderer()  # type: PeopleMaskRenderer
        self.__port = port                                  # type: int
        self.__post_client_hook = post_client_hook          # type: Optional[Callable[[], None]]

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the service."""
        # Set up the server socket.
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # type: socket.SocketType
        server_sock.bind(("127.0.0.1", self.__port))

        # Repeatedly:
        while True:
            # Listen for a connection.
            server_sock.listen(1)
            print("Listening for a connection on 127.0.0.1:{}...".format(self.__port))

            client_sock = None  # type: Optional[socket.SocketType]

            while client_sock is None:
                timeout = 0.1  # type: float
                readable, _, _ = select([server_sock], [], [], timeout)

                for s in readable:
                    if s is server_sock:
                        client_sock, client_endpoint = server_sock.accept()
                        print("Accepted connection from client @ {}".format(client_endpoint))

            # Once a client has connected, process any detection requests received from it. If the client
            # disconnects, keep the service running and loop back round to wait for another client.
            connection_ok = True            # type: bool
            intrinsics = None               # type: Optional[Tuple[float, float, float, float]]
            people_mask = None              # type: Optional[np.ndarray]
            receiver = RGBDFrameReceiver()  # type: RGBDFrameReceiver
            skeletons = None                # type: Optional[List[Skeleton3D]]

            while connection_ok:
                # First, try to read a control message from the client.
                control_msg = SkeletonControlMessage()  # type: SkeletonControlMessage
                connection_ok = SocketUtil.read_message(client_sock, control_msg)

                # If that succeeds:
                if connection_ok:
                    # Extract the control message value to tell us whether this is the start or end of a detection.
                    value = control_msg.extract_value()  # type: int

                    # If this is the start of a detection:
                    if value == SkeletonControlMessage.BEGIN_DETECTION:
                        if self.__debug:
                            print("Begin Detection")

                        # Try to read a frame header message.
                        max_images = 2  # type: int
                        header_msg = FrameHeaderMessage(max_images)  # type: FrameHeaderMessage
                        connection_ok = SocketUtil.read_message(client_sock, header_msg)

                        # If that succeeds:
                        if connection_ok:
                            # Set up a frame message accordingly.
                            image_shapes = header_msg.get_image_shapes()              # type: List[Tuple[int, int, int]]
                            image_byte_sizes = header_msg.get_image_byte_sizes()      # type: List[int]
                            frame_msg = FrameMessage(image_shapes, image_byte_sizes)  # type: FrameMessage

                            # Try to read the contents of the frame message from the client.
                            connection_ok = SocketUtil.read_message(client_sock, frame_msg)

                            # If that succeeds:
                            if connection_ok:
                                # Send an acknowledgement to the client.
                                connection_ok = SocketUtil.write_message(client_sock, AckMessage())

                                # Decompress the frame as necessary.
                                decompressed_frame_msg = frame_msg  # type: FrameMessage
                                if self.__frame_decompressor is not None:
                                    decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                                # Detect any people who are present in the frame.
                                receiver(decompressed_frame_msg)

                                skeletons, people_mask = self.__frame_processor(
                                    receiver.get_frame_index(), receiver.get_rgb_image(), receiver.get_depth_image(),
                                    receiver.get_pose(), intrinsics
                                )

                                # If the skeleton detector did not return a people mask, render one using the
                                # detected skeletons.
                                if people_mask is None:
                                    height, width = receiver.get_rgb_image().shape[:2]
                                    people_mask = self.__people_mask_renderer.render_people_mask(
                                        SkeletonDetectionService.__render_person_mask, skeletons,
                                        receiver.get_pose(), intrinsics, width, height
                                    )

                    # Otherwise, if this is the end of a detection:
                    elif value == SkeletonControlMessage.END_DETECTION:
                        if self.__debug:
                            print("End Detection")

                        # Assuming the skeletons have previously been detected (if not, it's because the client has
                        # erroneously called end_detection prior to begin_detection):
                        if skeletons is not None:
                            # Send them across to the client.
                            # noinspection PyTypeChecker
                            data = np.frombuffer(bytes(repr(skeletons), "utf-8"), dtype=np.uint8)  # type: np.ndarray
                            data_msg = DataMessage(len(data))  # type: DataMessage
                            np.copyto(data_msg.get_data(), data)

                            mask_msg = BinaryMaskMessage(people_mask.shape)  # type: BinaryMaskMessage
                            mask_msg.set_mask(people_mask)

                            connection_ok = \
                                SocketUtil.write_message(client_sock, SimpleMessage[int](int, len(data))) and \
                                SocketUtil.write_message(client_sock, data_msg) and \
                                SocketUtil.write_message(client_sock, mask_msg)

                            # Now that we've sent the skeletons, clear them so that they don't get sent to the client
                            # again erroneously in future frames.
                            skeletons = None
                            people_mask = None

                    # Otherwise, if the calibration is being set:
                    elif value == SkeletonControlMessage.SET_CALIBRATION:
                        if self.__debug:
                            print("Set Calibration")

                        # Try to read a calibration message.
                        calib_msg = CalibrationMessage()  # type: CalibrationMessage
                        connection_ok = SocketUtil.read_message(client_sock, calib_msg)

                        # If that succeeds:
                        if connection_ok:
                            # Store the camera intrinsics for later.
                            intrinsics = calib_msg.get_intrinsics()[0]

                            # Send an acknowledgement message.
                            connection_ok = SocketUtil.write_message(client_sock, AckMessage())

            # If there's a hook function to call after a client disconnects, call it.
            if self.__post_client_hook is not None:
                self.__post_client_hook()

    # PRIVATE STATIC METHODS

    @staticmethod
    def __render_person_mask(skeleton: Skeleton3D) -> None:
        """
        Render a person mask for a skeleton by rendering the skeleton's bounding shapes in white.

        :param skeleton:    The skeleton.
        :return:            The person mask.
        """
        glColor3f(1.0, 1.0, 1.0)
        SkeletonRenderer.render_bounding_shapes(skeleton)
