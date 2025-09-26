#!/usr/bin/env python3
"""
manager_server.py - Creates and manages the shared queue server.
This script must be run first to establish the Manager server.
"""

import multiprocessing as mp
import queue
import time
from multiprocessing.managers import BaseManager
import os
import signal
import sys
from queue import Queue
from contextlib import contextmanager
from typing import Optional

DEFAULT_IP = "localhost"
DEFAULT_PORT = 50000
DEFAULT_AUTHKEY = 'shared_queue_key'


class QueueManager(BaseManager):
    """Custom Manager class to handle our shared queue."""
    pass


def create_shared_queue(maxsize):
    """Factory function to create the shared queue."""
    if maxsize is None:
        kwargs = {}
    else:
        kwargs = dict(maxsize=maxsize)
    return Queue(**kwargs)


def connect_to_queue(ip=DEFAULT_IP, port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY):
    """Connect to the shared queue managed by the server."""
    if isinstance(authkey, str):
        authkey = authkey.encode()
    try:
        # Register the queue getter (no callable needed for client)
        QueueManager.register('get_queue')

        print(f"Connecting to queue {ip}:{port}")

        # Connect to the manager server
        manager = QueueManager(address=(ip, port), authkey=authkey)
        manager.connect()

        # Get the shared queue
        queue = manager.get_queue()
        return queue, manager

    except Exception as e:
        print(f"Failed to connect to manager server: {e}")
        print("Make sure manager_server.py is running first!")
        return None, None


@contextmanager
def queue_manager_server(ip, port, authkey, queue_maxsize):
    """Context manager for the QueueManager server."""
    print(f"Manager Server starting (PID: {os.getpid()})")

    # 1. Create the single, shared queue instance first.
    shared_queue_instance = create_shared_queue(queue_maxsize)

    # 2. Register a callable that ALWAYS returns the exact same queue instance.
    QueueManager.register('get_queue', callable=lambda: shared_queue_instance)

    if isinstance(authkey, str):
        authkey = authkey.encode()
    # Create the manager
    manager = QueueManager(address=(ip, port), authkey=authkey)
    server = manager.get_server()

    try:
        print(f"Manager server started on {ip}:{port}")
        print(f"Authkey: {authkey.decode()}")
        print(f"Queue maxsize: {queue_maxsize}")
        print("\nWaiting for client connections...")
        print("Press Ctrl+C to stop the server\n")
        yield server, shared_queue_instance  # Yield the actual shared queue
    finally:
        print("Initiating server cleanup...")
        try:
            # Drain the queue (optional)
            while not shared_queue_instance.empty():
                try:
                    shared_queue_instance.get_nowait()
                    print("Removed an item from the queue")
                except queue.Empty:
                    break
            print("Queue cleared")

            # The manager server is shut down by the signal handler
            # and the serve_forever() method's finally block.
            # manager.shutdown() is only for managers started with
            # manager.start() and is not needed here.
            print("Manager server shut down")
        except Exception as e:
            print(f"Error during server cleanup: {e}")


def signal_handler(signum, frame):
    """Handle Ctrl+C or SIGTERM gracefully."""
    print("\nReceived signal, shutting down via context manager...")
    # Let the context manager handle cleanup by exiting
    sys.exit(0)


@contextmanager
def queue_connection(ip=DEFAULT_IP, port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY):
    """Context manager for connecting to the shared queue."""

    q, m = connect_to_queue(ip, port, authkey)
    if q is None or m is None:
        print("Failed to connect to queue. Exiting.")
        raise RuntimeError("Queue connection failed")

    try:
        yield q
    finally:
        try:
            # No explicit shutdown or close needed; rely on garbage collection
            q._manager = None  # Break reference to manager
            del m  # Dereference manager to allow cleanup
            print("Client manager connection cleanup complete")
        except Exception as e:
            print(f"Error during client cleanup: {e}")


def get_all_available_items(q: mp.Queue):
    """
    Gets all items from a queue.
    If the queue is empty, it waits for the first item.
    Then, it retrieves all other items without blocking.
    """
    items = []
    try:
        # Block until the first item is available.
        items.append(q.get())

        # Then, get all other available items without blocking.
        while True:
            items.append(q.get_nowait())
            time.sleep(0.001)
    except queue.Empty:
        # The queue is now empty, return the collected items.
        return items
    return items


def main(ip:str=DEFAULT_IP, port:int=DEFAULT_PORT, authkey:str=DEFAULT_AUTHKEY,
         queue_maxsize: Optional[int]=None):
    """

    :param ip:
    :param port:
    :param authkey: a password to use the queue
    :param queue_maxsize:
    :return:
    """
    try:
        # Set up signal handlers to allow context manager cleanup
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Use the context manager to run the server
        with queue_manager_server(ip, port, authkey, queue_maxsize) as (server, queue):
            server.serve_forever()
    except Exception as e:
        print(f"Error starting manager server: {e}")
        return 1

    return 0



if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    exit(parse_function_and_call(main))