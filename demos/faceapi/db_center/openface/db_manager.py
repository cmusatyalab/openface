# -*- coding: UTF-8 -*-

"""
@file db_manager.py
@brief
    Implement of database manager in db_center.

Created on: 2016/1/14
"""

import os
import sqlite3
import logging

import faceapi
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.db_center import DbManager

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


_DB_FILE = os.path.join(faceapi.BASE_DIR, "data", "db.db3")
_SQL_CMD_CREATE_TAB = "CREATE TABLE IF NOT EXISTS "
_SQL_TABLE_FACE = (
                    "face_table(hash TEXT PRIMARY KEY, name TEXT, "
                    "representation TEXT, img_path TEXT)")
_SQL_GET_ALL_FACE = "SELECT * FROM face_table"
_SQL_ADD_FACE = (
                "INSERT or REPLACE INTO "
                "face_table(hash, name, representation, img_path) "
                "VALUES(?, ?, ?, ?)")


"""
.d8888b.  888
d88P  Y88b 888
888    888 888
888        888  8888b.  .d8888b  .d8888b
888        888     "88b 88K      88K
888    888 888 .d888888 "Y8888b. "Y8888b.
Y88b  d88P 888 888  888      X88      X88
 "Y8888P"  888 "Y888888  88888P'  88888P'
 """


class DbManagerOpenface(DbManager):
    def __init__(self, db_file_path=_DB_FILE):
        super(DbManagerOpenface, self).__init__(db_file_path)
        self._db_file = db_file_path
        self._logger = log_center.make_logger(__name__, logging.INFO)

        dir = os.path.dirname(db_file_path)
        if not os.path.exists(dir):
            os.makedirs(dir)

        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cur.execute(_SQL_CMD_CREATE_TAB + _SQL_TABLE_FACE)
                db.commit()
        except sqlite3.Error as e:
            self._logger.error(str(e))

    def dbList(self):
        rows = []
        try:
            db = sqlite3.connect(self._db_file)
            cur = db.cursor()
            cur.execute(_SQL_GET_ALL_FACE)

            columns = [column[0] for column in cur.description]
            for row in cur.fetchall():
                rows.append(dict(zip(columns, row)))
        except sqlite3.Error as e:
            self._logger.error(str(e))
            raise e
        finally:
            db.commit()
            db.close()

        return rows

    def addList(self, record_list):
        if type(record_list) is not list:
            self._logger.error("record_list is not a list type, do nothing.")
            return

        try:
            db = sqlite3.connect(self._db_file)
            cur = db.cursor()
        except sqlite3.Error as e:
            self._logger.error(str(e))
            raise exceptions.LibError(str(e))

        sql_add_list = []
        for record in record_list:
            info = (
                record.hash, record.person, record.eigen, record.img_path)
            self._logger.debug("add: " + str(info))
            sql_add_list.append(info)

        try:
            cur.executemany(_SQL_ADD_FACE, sql_add_list)
        except sqlite3.Error as e:
            self._logger.error(str(e))

        db.commit()
        db.close()
