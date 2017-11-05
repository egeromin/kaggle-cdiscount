"""
Utils to read the cdiscount data from mongodb

As we don't want to load everything into memory, only load
a subset, assuming that the BSON file has been imported 
into mongodb already.
"""
import os
from io import BytesIO

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mongo_helper import MongoHelper


class CDiscountDataset(Dataset):
    """
    A dataset loader for the mongodb CDiscount data
    """

    def __init__(self, data_folder, train=True, transform=None, mongo_db_name='cdiscount'):
        self.transform = transform
        path_categories = os.path.join(data_folder, "category_names.csv")
        path_samples = os.path.join(data_folder, "samples.jl")

        self.mongo_helper = MongoHelper(path_categories, db_name=mongo_db_name)
        self.mongo_helper.load_samples(path_samples)
        self.train = train
        self.ids = self.mongo_helper.get_ids(train)

        np.random.seed(5)  # generate random photo nums for each id
        self.photo_nums = (np.random.rand(len(self.ids)) * 4.0).astype(np.uint32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        doc_id = self.ids[item]
        photo_num = self.photo_nums[item]
        picture_data, label = self.mongo_helper.get_photo_data(
            doc_id, photo_num, train=self.train
        )
        img = Image.open(BytesIO(picture_data))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
