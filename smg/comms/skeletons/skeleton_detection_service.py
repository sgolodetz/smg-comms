import socket
import threading

from select import select
from typing import Callable, Optional

from ..base import FrameMessage


class SkeletonDetectionService:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, port: int = 7852, *,
                 frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = None):
        """
        TODO

        :param port:                TODO
        :param frame_decompressor:  TODO
        """
        self.__frame_decompressor: Optional[Callable[[FrameMessage], FrameMessage]] = frame_decompressor
        self.__port: int = port
        self.__service_thread: threading.Thread = threading.Thread(target=self.__run_service)
        self.__should_terminate: threading.Event = threading.Event()

        self.__lock: threading.Lock = threading.Lock()

    # DESTRUCTOR

    def __del__(self):
        """Destroy the service."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the service's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy the service at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def start(self) -> None:
        """Start the service."""
        self.__service_thread.start()

    def terminate(self) -> None:
        """Tell the service to terminate."""
        with self.__lock:
            if not self.__should_terminate.is_set():
                self.__should_terminate.set()
                self.__service_thread.join()

    # PRIVATE METHODS

    def __run_service(self) -> None:
        """Run the service."""
        # Set up the server socket and listen for connections.
        server_sock: socket.SocketType = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", self.__port))
        server_sock.listen(5)

        print(f"Listening for connections on 127.0.0.1:{self.__port}...")

        client_sock: Optional[socket.SocketType] = None

        while not self.__should_terminate.is_set():
            if client_sock is not None:
                pass
            else:
                timeout: float = 0.1
                readable, _, _ = select([server_sock], [], [], timeout)
                if self.__should_terminate.is_set():
                    break

                for s in readable:
                    if s is server_sock:
                        client_sock, client_endpoint = server_sock.accept()
                        print(f"Accepted connection from client @ {client_endpoint}")
