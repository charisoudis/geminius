"""
Author: Thanos Charisoudis (thanos@charisoudis.com)
License: MIT
"""
from __future__ import annotations

import logging
import os
import pathlib
import random
import re
import string
from pathlib import Path
from typing import Optional

from colorlog import ColoredFormatter
from dotenv import dotenv_values

GLOBAL_LOGGER = None
GLOBAL_ENV = None


class Env(dict):
    """Env Class:
    A class that represents the environment variables of the project. It is a dictionary of the environment variables
    defined in the `.env` file in the project root directory.
    """

    def __init__(self, env_path: Optional[Path] = None):
        if env_path is None:
            env_path = pathlib.Path(__file__).parent .parent / '.env'
        super().__init__(dotenv_values(env_path))
        for key, value in os.environ.items():
            if key not in self.keys():
                self.__setitem__(key, value)

    def __getitem__(self, item: str) -> str or dict:
        item_parts = map(lambda x: x.strip(), item.split(','))
        env_key = '_'.join(item_parts).upper()
        if env_key not in self.keys():
            # Return a sub-dictionary with key-value pairs where key start with env_key
            return {k[len(env_key) + 1:].lower(): v for k, v in self.items() if k.startswith(env_key)}
        return super().__getitem__(env_key)

    def __getattr__(self, item):
        if not hasattr(super(dict, self), item):
            return self.__getitem__(item)

    def __setattr__(self, key, value):
        os.environ[key] = value
        self.__setitem__(key, value)


class Logger(logging.Logger):
    """ Logger Class:
    The main logger of the project. It is a utility class to log colorfully messages to console.
    """

    LOG_FORMAT_DEFAULT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"

    def __init__(self, log_level: str = 'debug', log_format: str = LOG_FORMAT_DEFAULT, name: Optional[str] = None):
        """ Logger class constructor.

        Parameters
        ----------
        log_level: str
            debug Level (one of 'info', 'debug', 'warning', 'error', 'critical')
        log_format: str, optional
            log format or empty to use the default one
        name: str
            logger name (enables logs grouping/isolation)
        """
        self._stream = logging.StreamHandler()
        if name is None:
            name = Str.random(length=10).__str__()
        super().__init__(
            name=name,
            level=log_level.upper()
        )
        self._formatter = ColoredFormatter(log_format)

        self._log_level = None
        self._log_format = None

        self.log_level = log_level
        self.log_format = log_format

        self.addHandler(self._stream)

    @property
    def log_level(self):
        return self._log_level

    @log_level.setter
    def log_level(self, log_level: str):
        log_level = log_level.upper()
        if self._log_level == log_level:
            return

        # Update internal param
        self._log_level = log_level
        # Set log level
        logging.root.setLevel(log_level)
        self._stream.setLevel(log_level)
        self.setLevel(log_level)

    @property
    def log_format(self):
        return self._log_format

    @log_format.setter
    def log_format(self, log_format: str):
        if self._log_format == log_format:
            return
        # Update private param
        self._log_format = log_format
        # Set log format
        new_formatter = ColoredFormatter(log_format)
        self._formatter = new_formatter
        self._stream.setFormatter(new_formatter)


def get_global_env() -> Env:
    global GLOBAL_ENV
    if GLOBAL_ENV is None:
        GLOBAL_ENV = Env()
    return GLOBAL_ENV


def env_get(key: str, default: Optional[str] = None) -> str or dict:
    global_env = get_global_env()
    env_value = None
    try:
        for part in key.split('.'):
            if env_value is None:
                env_value = global_env[part]
            else:
                env_value = env_value[part]
        return env_value
    except KeyError:
        return default


def env_set(key: str, value: str) -> None:
    global_env = get_global_env()
    return global_env.__setattr__(key, value)


def get_global_logger() -> Logger:
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        global_log_level = env_get('LOG_LEVEL', 'debug')
        GLOBAL_LOGGER = Logger(name='global', log_level=global_log_level)
    return GLOBAL_LOGGER


def log(message: str, level: str = 'debug', logger: Optional[Logger] = None, **kwargs) -> None:
    if logger is None:
        logger = get_global_logger()
    if env_get('LOG', 'true').lower() == 'false':
        return
    return getattr(logger, level)(message, **kwargs)


class Str(str):
    def append(self, other: str) -> Str:
        return Str(str(self) + other)

    def append_if(self, other: str) -> Str:
        if not self.endswith(other):
            return self.append(other)

    def camel(self) -> Str:
        """Converts a string to camelCase."""
        pascal = self.pascal().__str__()
        return Str(pascal[0].lower() + pascal[1:])

    def lizard(self) -> Str:
        """Converts a string to lizardcase."""
        return self.snake().replace('_', '')

    def lower(self) -> Str:
        """Converts a string to lowercase."""
        return Str(super().lower())

    def pascal(self) -> Str:
        """Converts a string to PascalCase."""
        return Str(''.join(i.capitalize() for i in self.lower().split("_")))

    def replace(self, haystack: str, needle: str = '', count: int = -1) -> Str:
        return Str(re.sub(haystack, needle, self))

    def scanf(self, regex: str) -> list:
        return list(re.match(regex, self).groups())

    def snake(self) -> Str:
        return Str(re.sub(r'(?<!^)(?=[A-Z])', '_', self).lower())

    def split(self, regex: Optional[str] = None, maxsplit: int = -1) -> list:
        if regex is None:
            regex = ' '
        return list(re.split(regex, self))

    def trim(self) -> Str:
        return Str(self.strip())

    def upper(self) -> Str:
        return Str(super().upper())

    @staticmethod
    def random(length: int) -> Str:
        """ Get a random string containing ASCII alphanumerical characters.

        Parameters
        ----------
        length: int
            length of the generated string

        Returns
        -------
        Str
            random string with length equal to :attr:`length` containing random characters.
        """
        return Str(''.join(random.choices(string.ascii_letters + string.digits, k=length)))


if __name__ == '__main__':
    print(get_global_env())
    print(env_get('CLOUD_ML_PROJECT_ID'))

    # logger_ = Logger()
    # logger_.debug('This is a debug message')
    # logger_.info('This is an info message')
    # logger_.warning('This is an warning message')
    # logger_.error('This is an error message')
    # logger_.critical('This is a critical message')

    log('This is a debug message', level='debug')
    log('This is an info message', level='info')
    log('This is an warning message', level='warning')
    log('This is an error message', level='error')
    log('This is a critical message', level='critical')
