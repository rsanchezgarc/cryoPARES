#!/usr/bin/env python3
"""
manager_server.py - Creates and manages the shared queue server.
This script must be run first to establish the Manager server.

A single server instance can host **multiple independent queues** distinguished
by name.  Queues are created on demand the first time a client requests a given
name, so the server does not need to know queue names in advance.
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
DEFAULT_QUEUE_NAME = "default"

# Methods exposed on the proxy returned by get_queue().
# Must be declared explicitly so BaseProxy.__getattr__ allows them.
_QUEUE_EXPOSED = ('put', 'get', 'get_nowait', 'put_nowait', 'empty', 'qsize', 'task_done', 'join')


class QueueManager(BaseManager):
    """Custom Manager class to handle our shared queues."""
    pass


def _make_queue_getter(maxsize: Optional[int]):
    """
    Return a ``(get_queue, store)`` pair.

    ``get_queue(name)`` is the server-side callable registered with
    :class:`QueueManager`.  It lazily creates a new :class:`queue.Queue` the
    first time a given *name* is requested and returns the same instance on
    subsequent calls.  ``store`` is the underlying ``{name: Queue}`` dict,
    exposed so the server context manager can drain queues on shutdown.

    :param maxsize: Maximum size for each queue (``None`` = unlimited).
    :return: Tuple of ``(get_queue callable, queue store dict)``.
    """
    store: dict = {}

    def get_queue(name: str = DEFAULT_QUEUE_NAME) -> Queue:
        if name not in store:
            kwargs = {} if maxsize is None else {"maxsize": maxsize}
            store[name] = Queue(**kwargs)
            print(f"Created new queue '{name}'")
        return store[name]

    return get_queue, store


def _register_get_queue(callable_=None):
    """
    Register (or re-register) ``get_queue`` on :class:`QueueManager`.

    ``exposed`` is always set explicitly so that the returned proxy reliably
    exposes ``put``, ``get``, etc. regardless of Python version heuristics.

    Client-side calls (``callable_=None``) skip re-registration when
    ``get_queue`` is already registered — this prevents in-process tests
    from accidentally overwriting the server's callable with ``None``.

    :param callable_: Server-side callable (``None`` for client-only registration).
    """
    if callable_ is not None or 'get_queue' not in QueueManager._registry:
        QueueManager.register(
            'get_queue',
            callable=callable_,
            exposed=_QUEUE_EXPOSED,
        )


def connect_to_queue(ip=DEFAULT_IP, port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY,
                     queue_name: str = DEFAULT_QUEUE_NAME):
    """
    Connect to a named queue on the shared queue server.

    :param ip: IP address of the queue manager server.
    :param port: Port of the queue manager server.
    :param authkey: Authentication key for the queue manager server.
    :param queue_name: Name of the queue to connect to.  The server creates the
        queue on demand if it does not yet exist.
    :return: Tuple of ``(queue proxy, manager)`` or ``(None, None)`` on failure.
    """
    if isinstance(authkey, str):
        authkey = authkey.encode()
    try:
        _register_get_queue()   # client-side: no callable needed

        print(f"Connecting to queue '{queue_name}' at {ip}:{port}")

        manager = QueueManager(address=(ip, port), authkey=authkey)
        manager.connect()

        q = manager.get_queue(queue_name)
        return q, manager

    except Exception as e:
        print(f"Failed to connect to manager server: {e}")
        print("Make sure manager_server.py is running first!")
        return None, None


@contextmanager
def queue_manager_server(ip, port, authkey, queue_maxsize):
    """
    Context manager that starts a :class:`QueueManager` server.

    The server hosts multiple independent named queues.  Queues are created
    lazily on the first client request for a given name.

    Yields ``(server, queue_store)`` where *queue_store* is the live
    ``{name: Queue}`` dict (useful for introspection in tests).

    :param ip: IP address to bind the server to.
    :param port: Port to bind the server to.
    :param authkey: Authentication key for the server.
    :param queue_maxsize: Maximum size for each queue (``None`` = unlimited).
    """
    print(f"Manager Server starting (PID: {os.getpid()})")

    get_queue_fn, queue_store = _make_queue_getter(queue_maxsize)
    _register_get_queue(callable_=get_queue_fn)  # server-side: callable required

    if isinstance(authkey, str):
        authkey = authkey.encode()
    manager = QueueManager(address=(ip, port), authkey=authkey)
    server = manager.get_server()

    try:
        print(f"Manager server started on {ip}:{port}")
        print(f"Authkey: {authkey.decode()}")
        print(f"Queue maxsize: {queue_maxsize}")
        print("\nWaiting for client connections...")
        print("Press Ctrl+C to stop the server\n")
        yield server, queue_store
    finally:
        print("Initiating server cleanup...")
        try:
            for name, q in queue_store.items():
                drained = 0
                while not q.empty():
                    try:
                        q.get_nowait()
                        drained += 1
                    except queue.Empty:
                        break
                if drained:
                    print(f"Removed {drained} item(s) from queue '{name}'")
            print("All queues cleared")
            print("Manager server shut down")
        except Exception as e:
            print(f"Error during server cleanup: {e}")


def signal_handler(signum, frame):
    """Handle Ctrl+C or SIGTERM gracefully."""
    print("\nReceived signal, shutting down via context manager...")
    sys.exit(0)


@contextmanager
def queue_connection(ip=DEFAULT_IP, port=DEFAULT_PORT, authkey=DEFAULT_AUTHKEY,
                     queue_name: str = DEFAULT_QUEUE_NAME):
    """
    Context manager for connecting to a named queue on the shared server.

    :param ip: IP address of the queue manager server.
    :param port: Port of the queue manager server.
    :param authkey: Authentication key for the queue manager server.
    :param queue_name: Name of the queue to connect to.  The server creates the
        queue on demand if it does not yet exist.
    """
    q, m = connect_to_queue(ip, port, authkey, queue_name)
    if q is None or m is None:
        raise RuntimeError("Queue connection failed")

    try:
        yield q
    finally:
        try:
            q._manager = None  # Break reference to manager
            del m
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
        return items
    return items


def main(ip: str = DEFAULT_IP, port: int = DEFAULT_PORT, authkey: str = DEFAULT_AUTHKEY,
         queue_maxsize: Optional[int] = None):
    """
    Start the queue manager server.

    A single server can host multiple independent named queues on the same port.
    Queues are created on demand when first requested by a client.

    :param ip: IP address to bind the server to.
    :param port: Port to bind the server to.
    :param authkey: Authentication key (password) for the server.
    :param queue_maxsize: Maximum number of items per queue (``None`` = unlimited).
    """
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        with queue_manager_server(ip, port, authkey, queue_maxsize) as (server, _queue_store):
            server.serve_forever()
    except Exception as e:
        print(f"Error starting manager server: {e}")
        return 1

    return 0


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    exit(parse_function_and_call(main))