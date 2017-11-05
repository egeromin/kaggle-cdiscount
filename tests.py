"""
All unittests
"""

import unittest
from io import BytesIO

import numpy as np
from PIL import Image
from pymongo import MongoClient
from torch import FloatTensor
from torchvision import transforms

from dataset import CDiscountDataset
from mongo_helper import MongoHelper
from settings import setup_logging

setup_logging()


class TestDB(unittest.TestCase):

    def setUp(self):
        self.test_db_name = 'test-cdiscount'
        self.client = MongoClient()
        self.client.drop_database(self.test_db_name)
        self.db = self.client[self.test_db_name]

        example_pic_png = self.make_random_png()
        # use PNG instead of cdiscount's JPEG for testing:
        # JPEG is not exact and so after saving and reloading,
        # we get a different result.

        self.db.train.insert_many([{'_id': i,
                                    'imgs': [{"picture": example_pic_png}],
                                    'category_id': 'cat'} for i in range(10)])

    def make_random_png(self):
        np.random.seed(3)
        self.example_pic_data = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        example_pic = Image.fromarray(self.example_pic_data)
        example_pic_fh = BytesIO()
        example_pic.save(example_pic_fh, 'png')
        example_pic_fh.seek(0)
        return example_pic_fh.read()

    def tearDown(self):
        self.client.drop_database(self.test_db_name)


class TestMongoHelper(TestDB):

    def test_move_files(self):
        helper = MongoHelper('./test-data/category_names.csv',
                             db_name=self.test_db_name)
        helper.move_docs_to_val(range(5))
        self.assertEqual(self.db.train.count(), 5)
        self.assertEqual(self.db.val.count(), 5)


class TestDatasetLoader(TestDB):

    def test_load_dataset(self):
        dataset = CDiscountDataset('./test-data/', mongo_db_name=self.test_db_name)
        self.assertEqual(len(dataset), 10)
        img, label = dataset[0]
        img_data = np.array(img)
        self.assertTrue((img_data == self.example_pic_data).all())
        self.assertEqual(label, 'cat')

    def test_load_val_dataset(self):
        helper = MongoHelper('./test-data/category_names.csv',
                             db_name=self.test_db_name)
        helper.move_docs_to_val(range(5))
        dataset = CDiscountDataset('./test-data/', mongo_db_name=self.test_db_name,
                                   train=False)
        self.assertEqual(len(dataset), 5)
        img, label = dataset[0]
        img_data = np.array(img)
        self.assertTrue((img_data == self.example_pic_data).all())
        self.assertEqual(label, 'cat')

    def test_load_with_transform(self):
        transform = transforms.ToTensor()
        dataset = CDiscountDataset('./test-data/', mongo_db_name=self.test_db_name,
                                   transform=transform)
        self.assertEqual(len(dataset), 10)
        img, label = dataset[0]
        self.assertIsInstance(img, FloatTensor)
        img_data = (img.numpy() * 255).astype(np.uint8)

        example_pic_reshaped = np.rollaxis(self.example_pic_data, 2, 0)
        self.assertTrue((img_data == example_pic_reshaped).all())
        self.assertEqual(label, 'cat')
