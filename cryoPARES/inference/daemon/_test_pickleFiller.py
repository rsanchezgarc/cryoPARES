#!/usr/bin/env python3
"""
This illustrates another form of filling the queue
"""

import os
import pickle
import time
import glob
from queue import Empty
from contextlib import contextmanager
from multiprocessing.managers import BaseManager
from cryoPARES.inference.daemon.queueManager import DEFAULT_IP, DEFAULT_PORT, DEFAULT_AUTHKEY, queue_connection


POISON_PILL = None



def main(
        pickle_fname: str = "/home/sanchezg/cryo/data/preAlignedParticles/EMPIAR-10166/data/michael_input.pkl",
        ip: str = DEFAULT_IP,
        port: int = DEFAULT_PORT,
        authkey: str = DEFAULT_AUTHKEY,
):
    """
    Main function to monitor a directory and feed new .star files to the queue.

    :param pickle_fname: Fname with the pickle .star files
    :param ip: Queue manager IP address
    :param port: Queue manager port
    :param authkey: Queue manager authentication key
    :param pattern: File pattern to match
    :param check_interval: Seconds between directory checks
    """

    try:
        with queue_connection(ip=ip, port=port, authkey=authkey) as queue:
            with open(pickle_fname, "rb") as f:
                data_list = pickle.load(f)
                for item in data_list:
                    queue.put(item)
                print(f"Qsize: {queue.qsize()}")
    except KeyboardInterrupt:
        queue.put(POISON_PILL)
        print("Poison pill added")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    # queue.put(POISON_PILL)
    print("filler done!")
    return 0


if __name__ == "__main__":
    from argParseFromDoc import parse_function_and_call
    exit(parse_function_and_call(main))


    """
    
PYTHONPATH=. python cryoPARES/inference/daemon/spoolingFiller.py  ~/cryo/data/preAlignedParticles/EMPIAR-10166/data/projections/ --pattern "1000*star"
    
    """