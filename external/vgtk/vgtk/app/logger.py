
import os
import sys
import time
import logging
import numpy as np

# TODO: add tf.Summary suppport
import logging

class Logger():
    def __init__(self, log_file=None, log_level=logging.DEBUG):

        # create logger
        self.logger = logging.getLogger('my model')
        self.logger.setLevel(log_level)
        log_format = logging.Formatter("#%(asctime)s# %(message)s", 
                                       "%y-%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        self.logger.addHandler(console_handler)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_format)
            self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def log(self, scope, msg):
        self.logger.info(f'[{scope}] {msg}')

    def warning(self, scope, msg):
        self.logger.warning(f'[{scope}] {msg}')

    def debug(self, scope, msg):
        self.logger.debug(f'[{scope}] {msg}')

    def error(self, scope, msg):
        self.logger.error(f'[{scope}] {msg}')
