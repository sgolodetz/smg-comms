import numpy as np
import socket
import threading

from typing import Optional, TypeVar

from .message import Message


# TYPE VARIABLE

T = TypeVar('T', bound=Message)


# MAIN CLASS

class SocketUtil:
    """Utility functions related to sockets."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def read_message(sock: socket.SocketType, msg: T, stop_waiting: Optional[threading.Event] = None) -> bool:
        """
        Attempt to read a message of type T from the specified socket.

        :param sock:            The socket.
        :param msg:             The T into which to copy the message, if reading succeeded.
        :param stop_waiting:    An optional event that can be used to make the operation stop waiting if needed.
        :return:                True, if reading succeeded, or False otherwise.
        """
        try:
            data = b""  # type: bytes

            # Until we've read the number of bytes we were expecting:
            while len(data) < msg.get_size():
                try:
                    # Try to get the remaining bytes.
                    received = sock.recv(msg.get_size() - len(data))  # type: bytes

                    # If we made progress, append the new bytes to the buffer.
                    if len(received) > 0:
                        data += received

                    # Otherwise, something's wrong, so return False.
                    else:
                        return False
                except socket.timeout:
                    # If a timeout occurs and we should stop waiting, return False, else loop round for another try.
                    if stop_waiting is not None and stop_waiting.is_set():
                        return False

            # If we managed to get the number of bytes were were expecting, copy the buffer into the output message
            # and return True to indicate a successful read.
            np.copyto(msg.get_data(), np.frombuffer(data, dtype=np.uint8))
            return True
        except (ConnectionAbortedError, ConnectionResetError, ValueError):
            # If any (non-timeout) exceptions are thrown during the read, return False.
            return False

    @staticmethod
    def write_message(sock: socket.SocketType, msg: T) -> bool:
        """
        Attempt to write a message of type T to the specified socket.

        :param sock:    The socket.
        :param msg:     The message.
        :return:        True, if writing succeeded, or False otherwise.
        """
        try:
            sock.sendall(msg.get_data().tobytes())
            return True
        except (ConnectionAbortedError, ConnectionResetError):
            # If an exception is thrown during the write, return False.
            return False
