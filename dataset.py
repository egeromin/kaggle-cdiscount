"""
Utils to read the cdiscount data from mongodb

As we don't want to load everything into memory, only load
a subset, assuming that the BSON file has been imported 
into mongodb already.
"""
from torch.utils.data import Dataset


class CDiscountDataset(Dataset):
    """
    A dataset loader for the CDiscount data, stored in mongodb
    """

    def __init__(self, num_images=None, train=True, transform=None, mongo_conn=None):
        self.transform = transform
        self.mongo_db_name = "cdiscount"
        if train:
            self.mongo_collection_name = ''

