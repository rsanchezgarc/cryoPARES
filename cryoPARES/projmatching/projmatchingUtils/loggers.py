import sys
from functools import lru_cache
from logging import getLogger, StreamHandler, Formatter, INFO, DEBUG, WARN
import threading

class CustomStreamHandler(StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.stream_lock = threading.Lock()

    def emit(self, record):
        with self.stream_lock:
            try:
                msg = self.format(record)
                stream = self.stream
                # Add a prefix to easily identify the source of the log
                prefix = "[MAIN] " if record.name == "mainLogger" else "[WORKER] "
                stream.write(prefix + msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


@lru_cache(1)
def getWorkerLogger(verbose: bool):
    return _getLogger(verbose, "workerLogger")


@lru_cache(1)
def getMainLogger(verbose: bool):
    return _getLogger(verbose, "mainLogger")


def _getLogger(verbose, loggerName):
    logger = getLogger(name=loggerName)
    loggerLevel = INFO if verbose else WARN
    logger.setLevel(loggerLevel)

    # Use the custom StreamHandler
    handler = CustomStreamHandler(sys.stdout)
    handler.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
