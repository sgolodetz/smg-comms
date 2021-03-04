import numpy as np
import socket

from OpenGL.GL import *
from select import select
from typing import Callable, List, Optional, Tuple

from smg.opengl import OpenGLFrameBuffer, OpenGLMatrixContext, OpenGLUtil
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Skeleton, SkeletonRenderer

from ..base import *
from .skeleton_control_message import SkeletonControlMessage


class SkeletonDetectionService:
    """A skeleton detection service to which a single client can make detection requests over a network."""

    # CONSTRUCTOR

    def __init__(self, frame_processor: Callable[[np.ndarray, np.ndarray, np.ndarray], List[Skeleton]],
                 port: int = 7852, *, debug: bool = False,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        Construct a skeleton detection service.

        :param frame_processor:     The function to use to detect skeletons in frames.
        :param port:                The port on which the service should listen for a connection.
        :param debug:               Whether to print out debug messages.
        :param frame_decompressor:  An optional function to use to decompress received frames.
        """
        self.__debug: bool = debug
        self.__framebuffer: Optional[OpenGLFrameBuffer] = None
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__frame_processor: Callable[[np.ndarray, np.ndarray, np.ndarray], List[Skeleton]] = frame_processor
        self.__port: int = port

    # PUBLIC METHODS

    def run(self) -> None:
        """Run the service."""
        # Set up the server socket.
        server_sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))

        # Repeatedly:
        while True:
            # Listen for a connection.
            server_sock.listen(1)
            print(f"Listening for a connection on 127.0.0.1:{self.__port}...")

            client_sock: Optional[socket.SocketType] = None

            while client_sock is None:
                timeout: float = 0.1
                readable, _, _ = select([server_sock], [], [], timeout)

                for s in readable:
                    if s is server_sock:
                        client_sock, client_endpoint = server_sock.accept()
                        print(f"Accepted connection from client @ {client_endpoint}")

            # Once a client has connected, process any detection requests received from it. If the client
            # disconnects, keep the service running and loop back round to wait for another client.
            connection_ok: bool = True
            intrinsics: Optional[Tuple[float, float, float, float]] = None
            people_mask: Optional[np.ndarray] = None
            receiver: RGBDFrameReceiver = RGBDFrameReceiver()
            skeletons: Optional[List[Skeleton]] = None

            while connection_ok:
                # First, try to read a control message from the client.
                control_msg: SkeletonControlMessage = SkeletonControlMessage()
                connection_ok = SocketUtil.read_message(client_sock, control_msg)

                # If that succeeds:
                if connection_ok:
                    # Extract the control message value to tell us whether this is the start or end of a detection.
                    value: int = control_msg.extract_value()

                    # If this is the start of a detection:
                    if value == SkeletonControlMessage.BEGIN_DETECTION:
                        if self.__debug:
                            print("Begin Detection")

                        # Try to read a frame header message.
                        max_images: int = 2
                        header_msg: FrameHeaderMessage = FrameHeaderMessage(max_images)
                        connection_ok = SocketUtil.read_message(client_sock, header_msg)

                        # If that succeeds:
                        if connection_ok:
                            # Set up a frame message accordingly.
                            image_shapes: List[Tuple[int, int, int]] = header_msg.get_image_shapes()
                            image_byte_sizes: List[int] = header_msg.get_image_byte_sizes()
                            frame_msg: FrameMessage = FrameMessage(image_shapes, image_byte_sizes)

                            # Try to read the contents of the frame message from the client.
                            connection_ok = SocketUtil.read_message(client_sock, frame_msg)

                            # If that succeeds:
                            if connection_ok:
                                # Send an acknowledgement to the client.
                                connection_ok = SocketUtil.write_message(client_sock, AckMessage())

                                # Decompress the frame as necessary.
                                decompressed_frame_msg: FrameMessage = frame_msg
                                if self.__frame_decompressor is not None:
                                    decompressed_frame_msg = self.__frame_decompressor(frame_msg)

                                # Detect any people who are present in the frame.
                                receiver(decompressed_frame_msg)

                                skeletons = self.__frame_processor(
                                    receiver.get_rgb_image(), receiver.get_depth_image(), receiver.get_pose()
                                )

                                # Render the people mask.
                                height, width = receiver.get_rgb_image().shape[:2]
                                people_mask = self.__render_people_mask(
                                    skeletons, receiver.get_pose(), intrinsics, width, height
                                )
                                # import cv2
                                # cv2.imshow("People Mask", people_mask)
                                # cv2.waitKey(1)

                    # Otherwise, if this is the end of a detection:
                    elif value == SkeletonControlMessage.END_DETECTION:
                        if self.__debug:
                            print("End Detection")

                        # Assuming the skeletons have previously been detected (if not, it's because the client has
                        # erroneously called end_detection prior to begin_detection):
                        if skeletons is not None:
                            # Send them across to the client.
                            # noinspection PyTypeChecker
                            data: np.ndarray = np.frombuffer(bytes(repr(skeletons), "utf-8"), dtype=np.uint8)
                            data_msg: DataMessage = DataMessage(len(data))
                            np.copyto(data_msg.get_data(), data)

                            mask_msg: BinaryMaskMessage = BinaryMaskMessage(people_mask.shape)
                            mask_msg.set_mask(people_mask)

                            connection_ok = \
                                SocketUtil.write_message(client_sock, SimpleMessage[int](len(data))) and \
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
                        calib_msg: CalibrationMessage = CalibrationMessage()
                        connection_ok = SocketUtil.read_message(client_sock, calib_msg)

                        # If that succeeds:
                        if connection_ok:
                            # Store the camera intrinsics for later.
                            intrinsics = calib_msg.get_intrinsics()[0]

                            # Send an acknowledgement message.
                            connection_ok = SocketUtil.write_message(client_sock, AckMessage())

    # PRIVATE METHODS

    def __render_people_mask(self, skeletons: List[Skeleton], world_from_camera: np.ndarray,
                             intrinsics: Optional[Tuple[float, float, float, float]],
                             width: int, height: int) -> np.ndarray:
        """
        TODO

        :param skeletons:           TODO
        :param world_from_camera:   TODO
        :param intrinsics:          TODO
        :param width:               TODO
        :param height:              TODO
        :return:                    TODO
        """
        # If the camera intrinsics aren't available, early out.
        if intrinsics is None:
            return np.zeros((height, width), dtype=np.uint8)

        # If the OpenGL framebuffer hasn't been constructed yet, construct it now.
        # TODO: Support image size changes.
        if self.__framebuffer is None:
            self.__framebuffer = OpenGLFrameBuffer(width, height)

        # Render a mask of the skeletons' bounding shapes to the framebuffer.
        with self.__framebuffer:
            # Set the viewport to encompass the whole framebuffer.
            OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), (width, height))

            # Clear the background to black.
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, width, height
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(np.linalg.inv(world_from_camera))
                )):
                    # Render the skeletons' bounding shapes in white.
                    glColor3f(1.0, 1.0, 1.0)
                    for skeleton in skeletons:
                        SkeletonRenderer.render_bounding_shapes(skeleton)

                    # Make a binary mask from the contents of the framebuffer, and return it.
                    # TODO: Make this a function in OpenGLUtil.
                    buffer: bytes = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
                    mask: np.ndarray = np.frombuffer(
                        buffer, dtype=np.uint8
                    ).reshape((height, width, 3))[::-1, :]

                    return mask[:, :, 0]
