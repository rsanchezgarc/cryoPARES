#!/usr/bin/env python3
"""
spooler.py - Monitors a directory for new .star files and adds them to the shared queue for DaemonInferencer.
"""

import os
import time
import glob
from queue import Empty
from contextlib import contextmanager
from multiprocessing.managers import BaseManager
from cryoPARES.inference.daemon.queueManager import DEFAULT_IP, DEFAULT_PORT, DEFAULT_AUTHKEY, queue_connection


POISON_PILL = None

def monitor_directory(directory: str, pattern: str = "*.star", interval: int = 10):
    """
    Monitor a directory for new files matching the pattern and yield their paths.

    :param directory: Directory to monitor
    :param pattern: File pattern to match (default: *.star)
    :param interval: Time interval (seconds) between directory checks
    """
    processed_files = set()

    while True:
        current_files = set(glob.glob(os.path.join(directory, pattern)))
        new_files = current_files - processed_files

        for file_path in new_files:
            if os.path.isfile(file_path):
                yield file_path
                processed_files.add(file_path)

        time.sleep(interval)


def main(
        directory: str,
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        authkey: str = DEFAULT_AUTHKEY,
        pattern: str = "*.star",
        check_interval: int = 10
):
    """
    Main function to monitor a directory and feed new .star files to the queue.

    :param directory: Directory to monitor for .star files
    :param ip: Queue manager IP address
    :param port: Queue manager port
    :param authkey: Queue manager authentication key
    :param pattern: File pattern to match
    :param check_interval: Seconds between directory checks
    """

    try:
        with queue_connection(ip=ip, port=port, authkey=authkey) as queue:
            print(f"Spooler started, monitoring {directory} for {pattern} files...")

            for star_file in monitor_directory(directory, pattern, check_interval):
                print(f"Found new file: {star_file}")
                try:
                    queue.put(star_file)
                    print(f"Added {star_file} to queue")
                except Exception as e:
                    print(f"Error adding {star_file} to queue: {e}")
                print(f"Qsize: {queue.qsize()}")
    except KeyboardInterrupt:
        queue.put(POISON_PILL)
        print("Poison pill added")
    except Exception as e:
        print(f"Spooler error: {e}")
        return 1
    print("filler done!")
    return 0


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    exit(parse_function_and_call(main))


    """
    
PYTHONPATH=. python cryoPARES/inference/daemon/spoolingFiller.py  ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/ --pattern "1000*star"
    
    """