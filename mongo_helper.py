"""
Helper class to interact with cdiscount mongodb.

Assumes all CDiscount data has already bin loaded
into mongodb using mongorestore

In particular, helps move products from the original
'train' set to a validation set in a different
mongo collection
"""
from argparse import ArgumentParser
from functools import reduce
import operator

import logging
from pymongo import MongoClient, InsertOne, DeleteOne
import pandas as pd
import numpy as np
from pymongo.errors import BulkWriteError
from sklearn.externals import joblib

from settings import setup_logging


class MongoHelper:

    def __init__(self, path_categories, db_name='cdiscount'):
        self.client = MongoClient()
        self.db = self.client[db_name]
        self.train = self.db.train
        self.val = self.db.val
        self.categories = list(pd.read_csv(path_categories)["category_id"])
        print("Loaded {} categories".format(len(self.categories)))
        self.samples = None

    def select_document_samples(self, n_per_class):
        """
        Select document samples for each category by iterating
        over each sample in database until we have enough in  
        each category.
        
        Since the dataset is so big, this is the most efficient 
        method.
        """
        samples = {category: [] for category in self.categories}
        num_samples_per_cat = pd.Series(np.zeros(len(self.categories)),
                                        index=self.categories)
        for i, doc in enumerate(self.train.find(
                {}, {'_id': 1, 'category_id': 1}), 1):
            category = doc['category_id']
            doc_id = doc['_id']

            if num_samples_per_cat[category] < n_per_class:
                samples[category].append(doc_id)
                num_samples_per_cat[category] += 1

            if i % 1000 == 0:
                num_classes_missing = (num_samples_per_cat < n_per_class).sum()
                logging.info("Missing {} categories for completion".format(num_classes_missing))
                if num_classes_missing == 0:
                    break

        self.samples = samples
        return self.samples

    def _copy_docs_to_val(self, docs):
        """
        Insert a certain set of documents to val
        :param docs: The documents to insert
        :return: The documents, split into the written set
        and the remaining set
        """
        requests = [InsertOne(doc) for doc in docs]
        try:
            result = self.val.bulk_write(requests)
        except BulkWriteError as bwe:
            num_written = bwe.details['nInserted']
        else:
            num_written = result.bulk_api_result['nInserted']
        logging.info("Written {} docs to val".format(num_written))
        written = docs[:num_written]
        remaining = docs[num_written:]
        return written, remaining

    def _delete_docs(self, docs):
        """
        Delete a certain set of documents from train
        :param ids: The ids of the documents to remove
        :return: The ids of the documents *not* removed
        """
        requests = [DeleteOne(doc) for doc in docs]
        try:
            result = self.train.bulk_write(requests)
        except BulkWriteError as bwe:
            num_removed = bwe.details['nRemoved']
        else:
            num_removed = result.bulk_api_result['nRemoved']
        logging.info("Deleted {} docs from train".format(num_removed))
        remaining = docs[num_removed:]
        return remaining

    def move_docs_to_val(self, ids):
        """
        Method to move a list of documents from the 'train'
        collection to the 'val' collection
        """
        logging.info("Getting docs")
        docs = [self.train.find_one({'_id': doc_id}) for doc_id in ids]
        logging.info("Moving docs from train to val")
        while len(docs) > 0:
            written, docs = self._copy_docs_to_val(docs)
            while len(written) > 0:
                written = self._delete_docs(written)
        logging.info("Moved {} docs from train to val".format(len(ids)))

    def save_samples(self, path):
        """
        Method to save the samples as a scipy joblib object
        """
        joblib.dump(self.samples, path)

    def load_samples(self, path):
        """Load samples from joblib"""
        self.samples = joblib.load(path)
        logging.info("Loaded samples with {} categories "
                     "and minimum {} in each category".format(len(self.samples.keys()),
                                                              min([len(cat) for cat in self.samples.values()])))

    def get_ids(self, train=True):
        if train:
            return reduce(operator.add, self.samples.values(), [])
        return [doc['_id'] for doc in self.val.find({}, {'_id': 1})]

    def get_photo_data(self, doc_id, photo_num, train=True):
        """
        Get the photo data given a doc id and a photo number.
        This method is 'safe' in that the photo number does not
        have to be within bounds
        """
        db = self.val
        if train:
            db = self.train
        doc = db.find_one({'_id': doc_id})
        num_photos = len(doc['imgs'])
        photo_num = photo_num % num_photos  # 'safe' photo number
        return doc["imgs"][photo_num]["picture"]


def main():
    parser = ArgumentParser(description="Helper script to interact with cdiscount mongodb.")
    parser.add_argument("--cat", help="Path to categories",
                        default="./data/cdiscount/category_names.csv")
    parser.add_argument("--num", help="Number of samples to produce", type=int, default=100)
    parser.add_argument("--move", action='store_true', help="Move samples from train to val?")
    parser.add_argument("--save", help="Location to save image samples to",
                        default="./data/cdiscount/samples.jl")

    args = parser.parse_args()

    helper = MongoHelper(args.cat)
    samples = helper.select_document_samples(args.num)

    if args.move:
        ids = reduce(operator.add, samples.values(), [])
        helper.move_docs_to_val(ids)

    helper.save_samples(args.save)


if __name__ == "__main__":
    setup_logging()
    main()
