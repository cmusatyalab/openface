# -*- coding: UTF-8 -*-

"""
@file db_manager.py
@brief
    Implement of database manager in database.

Created on: 2016/1/14
"""

import os
import sqlite3
import logging

import faceapi
from faceapi import exceptions
from faceapi.utils import log_center
from faceapi.database import DbManager

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


_DB_FILE = os.path.join(faceapi.BASE_DIR, "data", "facedb.db3")
_SQL_CMD_CREATE_TAB = "CREATE TABLE IF NOT EXISTS "
_SQL_TABLE_FACE = (
    "face_table(hash TEXT PRIMARY KEY, "
    "name TEXT, "
    "eigen TEXT, "
    "src_hash TEXT, "
    "face_img TEXT, "
    "class_id INTEGER)")
_SQL_GET_ALL_FACE = "SELECT * FROM face_table"
_SQL_ROWS = "SELECT COUNT(*) FROM face_table"
_SQL_ADD_FACE = (
    "INSERT or REPLACE INTO "
    "face_table "
    "VALUES(?, ?, ?, ?, ?, ?)")
_SQL_GET_FACE_WITH_FIELD = "SELECT * FROM face_table WHERE {}={} LIMIT {}"
_SQL_DISTINCT_SEARCH = "select distinct {} from face_table order by {}"


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
    def __init__(self, db_file=_DB_FILE):
        super(DbManagerOpenface, self).__init__(db_file)
        self._db_file = db_file
        self._log = log_center.make_logger(__name__, logging.INFO)
        self._log.info("db_file: {}".format(db_file))

        dir = os.path.dirname(db_file)
        if not os.path.exists(dir):
            os.makedirs(dir)

        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cur.execute(_SQL_CMD_CREATE_TAB + _SQL_TABLE_FACE)
                db.commit()
        except sqlite3.Error as e:
            self._log.error(str(e))

    def count(self):
        rows = 0
        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cur.execute(_SQL_ROWS)
                # result = cur.fetchone()
                # rows = result[0]
                (rows, ) = cur.fetchone()
        except sqlite3.Error as e:
            self._log.error(str(e))
            raise e

        return rows

    def dbList(self):
        rows = []
        db = None
        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cur.execute(_SQL_GET_ALL_FACE)
                columns = [column[0] for column in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(columns, row)))
        except sqlite3.Error as e:
            self._log.error(str(e))
            raise e

        return rows

    def addList(self, record_list):
        if type(record_list) is not list:
            self._log.error("record_list is not a list type, do nothing.")
            return

        try:
            db = sqlite3.connect(self._db_file)
            cur = db.cursor()
        except sqlite3.Error as e:
            self._log.error(str(e))
            raise exceptions.LibError(str(e))

        sql_add_list = []
        for record in record_list:
            rep_str = ",".join(str(x) for x in record.eigen)
            info = (
                record.hash, record.name, rep_str,
                record.src_hash, record.face_img, record.class_id)
            self._log.debug("add: " + str(info))
            sql_add_list.append(info)

        try:
            cur.executemany(_SQL_ADD_FACE, sql_add_list)
        except sqlite3.Error as e:
            self._log.error(str(e))

        db.commit()
        db.close()

    def search(self, field, value, count):
        rows = []
        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cmd = _SQL_GET_FACE_WITH_FIELD.format(field, value, count)
                self._log.debug("sql cmd: {}".format(cmd))
                cur.execute(cmd)

                columns = [column[0] for column in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(columns, row)))
        except sqlite3.Error as e:
            self._log.error(str(e))
            raise e

        return rows

    def distinct_search(self, field_list, order_field):
        rows = []
        try:
            with sqlite3.connect(self._db_file) as db:
                cur = db.cursor()
                cmd = _SQL_DISTINCT_SEARCH.format(
                    ','.join(field_list), order_field)
                self._log.debug("sql cmd: {}".format(cmd))
                cur.execute(cmd)

                columns = [column[0] for column in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(columns, row)))
        except sqlite3.Error as e:
            self._log.error(str(e))
            raise e

        return rows
