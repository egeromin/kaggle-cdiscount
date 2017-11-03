"""
Utils to read the cdiscount data from mongodb

As we don't want to load everything into memory, only load
a subset, assuming that the BSON file has been imported 
into mongodb already.
"""
import os
import random

from torch.utils.data import Dataset

from mongo_helper import MongoHelper


class CDiscountDataset(Dataset):
    """
    A dataset loader for the CDiscount data, stored in mongodb
    """

    def __init__(self, data_folder, train=True, transform=None):
        self.transform = transform
        path_categories = os.path.join(data_folder, "category_names.csv")
        path_samples = os.path.join(data_folder, "samples.jl")
        self.mongo_helper = MongoHelper(path_categories)
        self.mongo_helper.load_samples(path_samples)
        self.train = train
        self.ids = self.mongo_helper.get_ids(train)
        random.seed(6)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        doc_id = self.ids[item]
        picture_data = self.mongo_helper.get_photo_data(doc_id, random.randint(0, 3))
        # todo : load data into image and then apply transform.
