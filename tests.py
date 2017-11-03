"""
All unittests
"""

import unittest

from pymongo import MongoClient

from mongo_helper import MongoHelper
from settings import setup_logging

setup_logging()


class TestMongoHelper(unittest.TestCase):

    def setUp(self):
        self.test_db_name = 'test-cdiscount'
        self.client = MongoClient()
        self.db = self.client[self.test_db_name]
        self.db.train.insert_many([{'_id': i, 'data': 'pumpkin'} for i in range(10)])

    def test_move_files(self):
        helper = MongoHelper('./data/cdiscount/category_names.csv',
                             db_name=self.test_db_name)
        helper.move_docs_to_val(range(5))
        self.assertEqual(self.db.train.count(), 5)
        self.assertEqual(self.db.val.count(), 5)

    def tearDown(self):
        self.client.drop_database(self.test_db_name)
