# -*- coding: UTF-8 -*-

"""
@file exceptions.py
@brief
    Get fb info from ftp server and manage the db.

Created on: 2013/12/4
"""


class NetworkError(Exception):

    def __init__(self, msg):
        super(NetworkError, self).__init__(msg)


class LibError(Exception):

    def __init__(self, msg):
        super(LibError, self).__init__(msg)
