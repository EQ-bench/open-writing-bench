import os
import logging
import sys

def setup_logging(verbosity: str):
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    level = levels.get(verbosity.upper(), logging.INFO)

    # Hard reconfigure: remove existing handlers, then add ours
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(fmt)

    root.addHandler(handler)
    root.setLevel(level)

    # Capture warnings and lower noisy libraries if desired
    logging.captureWarnings(True)


def get_verbosity(args_verbosity: str) -> str:
    if args_verbosity:
        return args_verbosity
    return os.getenv("LOG_VERBOSITY", "INFO")
