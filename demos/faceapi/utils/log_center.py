# -*- coding: UTF-8 -*-

"""
@file log_center.py
@brief
    Log tools with level.

Created on: 2013/12/4
"""

# import os
import logging
# from logging import StreamHandler

"""
print '\033[1;30mGray like Ghost\033[1;m'
print '\033[1;31mRed like Radish\033[1;m'
print '\033[1;32mGreen like Grass\033[1;m'
print '\033[1;33mYellow like Yolk\033[1;m'
print '\033[1;34mBlue like Blood\033[1;m'
print '\033[1;35mMagenta like Mimosa\033[1;m'
print '\033[1;36mCyan like Caribbean\033[1;m'
print '\033[1;37mWhite like Whipped Cream\033[1;m'
print '\033[1;38mCrimson like Chianti\033[1;m'
print '\033[1;41mHighlighted Red like Radish\033[1;m'
print '\033[1;42mHighlighted Green like Grass\033[1;m'
print '\033[1;43mHighlighted Brown like Bear\033[1;m'
print '\033[1;44mHighlighted Blue like Blood\033[1;m'
print '\033[1;45mHighlighted Magenta like Mimosa\033[1;m'
print '\033[1;46mHighlighted Cyan like Caribbean\033[1;m'
print '\033[1;47mHighlighted Gray like Ghost\033[1;m'
print '\033[1;48mHighlighted Crimson like Chianti\033[1;m'
"""

"""
8888888b.            .d888 d8b
888  "Y88b          d88P"  Y8P
888    888          888
888    888  .d88b.  888888 888 88888b.   .d88b.  .d8888b
888    888 d8P  Y8b 888    888 888 "88b d8P  Y8b 88K
888    888 88888888 888    888 888  888 88888888 "Y8888b.
888  .d88P Y8b.     888    888 888  888 Y8b.          X88
8888888P"   "Y8888  888    888 888  888  "Y8888   88888P'
 """

ORIGIN = 'ORIGIN'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
ENDC = '\033[0m'
HIGHLIGH_RED = '\033[0;41m'
HIGHLIGH_BULE = '\033[0;44m'

LOG_MSG_FORMAT = (
                "[%(levelname)s]@<%(module)s:%(funcName)s:%(lineno)d> "
                "%(message)s"
                )

NONE = 0
VERBOSE = 1
DEBUG = 2
ERROR = 3


def make_logger(logger_name, defualt_lavel=logging.ERROR):
    # logging.basicConfig(level=defualt_lavel, format=LOG_MSG_FORMAT)
    logger = logging.getLogger(logger_name)
    logger.setLevel(defualt_lavel)

    if not logger.handlers:
        out_handler = ColorOutputHandler()
        logger.addHandler(out_handler)

    return logger


class ColorOutputHandler(logging.StreamHandler):

    _level_map = {
        logging.DEBUG: (None, 'cyan', False),
        logging.INFO: (None, 'green', False),
        logging.WARNING: (None, 'purple', True),
        logging.ERROR: (None, 'red', True),
        logging.CRITICAL: ('red', 'white', True),
    }

    _color_map = {
        'black': 0,
        'red': 1,
        'green': 2,
        'yellow': 3,
        'blue': 4,
        'purple': 5,
        'cyan': 6,
        'white': 7
    }

    def __init__(self):
        super(ColorOutputHandler, self).__init__()

    # def emit(self, record):
    #     print "ColorOutputHandler:emit \n%s \n" % (record)
    #     record.levelno
    #     super(ColorOutputHandler, self).emit(record)

    def format(self, record):

        if record.levelno in self._level_map:
            bg, fg, bold = self._level_map[record.levelno]
        else:
            # Defaults
            bg = None
            fg = 'white'
            bold = False

        template = [
            self._get_color(None, "blue", None), "[",
            ENDC,
            self._get_color(bg, fg, bold),
            "%(levelname)s",
            ENDC,
            self._get_color(None, "blue", bold), "]", "@", "<",
            ENDC,
            self._get_color(None, "yellow", None),
            "%(module)s", ":", "%(funcName)s", ":", "%(lineno)d",
            ENDC,
            self._get_color(None, "blue", bold), "> ",
            ENDC,
            self._get_color(bg, fg, bold),
            "%(message)s",
            ENDC,
        ]

        format = "".join(template)

        formatter = logging.Formatter(format, None)
        output = formatter.format(record)

        # return super(ColorOutputHandler, self).format(record)
        return output

    def _get_color(self, bg=None, fg=None, bold=False):

        params = []
        # if bg in self._color_map:
        #    params.append('\e[{0}m'.format(40))
        if fg in self._color_map:
            params.append('\033[0{0}m'.format(self._color_map[fg] + 30))
        # if bold:
            # params.append('1')

        color_code = ''.join(params)
        # print color_code

        return color_code


# class LogLv:

#     def __init__(self, logLv=DEBUG):
#         self._logLv = logLv

#     def set_level(self, level):
#         self._logLv = level
#         return None

#     def verbose(self, tag, msg):
#         self.log(VERBOSE, tag, ORIGIN, msg, ORIGIN)
#         return None

#     def debug(self, tag, msg):
#         self.log(DEBUG, tag, HIGHLIGH_BULE, msg, BLUE)
#         return None

#     def error(self, tag, msg):
#         self.log(ERROR, tag, HIGHLIGH_RED, msg, RED)
#         return None

#     def log(self, level, tag, tagcolor, msg, msgcolor):
#         if self._logLv == NONE:
#             return

#         if level < self._logLv:
#             return

#         if tagcolor == ORIGIN:
#             print "[%s] %s" % (tag, msg)
#             return

#         print "%s[%s]%s %s%s%s" % (tagcolor, os.path.basename(__file__), ENDC, msgcolor, msg, ENDC)
#         return None
