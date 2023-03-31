import logging
import os
import sys


class PrintSilencer:
    def __enter__(self):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._devnull_open = open(os.devnull, 'w')
        sys.stdout = self._devnull_open
        sys.stderr = self._devnull_open
        logging.getLogger().disabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._devnull_open.close()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        logging.getLogger().disabled = True
