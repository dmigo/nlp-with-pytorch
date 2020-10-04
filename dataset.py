import pandas
from typing import Dict, Any
from torch.utils.data import Dataset
from pandas import DataFrame
from vectorizer import ReviewVectorizer

TRAIN = "train"
VALIDATION = "val"
TEST = "test"


class ReviewDataset(Dataset):
    def __init__(self, review_df: DataFrame, vectorizer: ReviewVectorizer):
        """
        Args:
            review_df (pandas.DataFrame): the dataset
            vectorizer (ReviewVectorizer): vectorizer instantiated from dataset
        """

        self.review_df = review_df
        self._vectorizer = vectorizer

        self.train_df = self.review_df[self.review_df.split == TRAIN]
        self.train_size = len(self.train_df)

        self.val_df = self.review_df[self.review_df.split == VALIDATION]
        self.val_size = len(self.val_df)

        self.test_df = self.review_df[self.review_df.split == TEST]
        self.test_size = len(self.test_df)

        self._splits = {
            TRAIN: (self.train_df, self.train_size),
            VALIDATION: (self.val_df, self.val_size),
            TEST: (self.test_df, self.test_size),
        }

        self.set_split(TRAIN)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, review_csv: str) -> Dataset:
        """Load dataset and make a new vectorizer from scratch

        Args:
            review_csv (str): location of the dataset
        Returns:
            an instance of ReviewDataset
        """
        review_df = pandas.read_csv(review_csv)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))

    def get_vectorizer(self) -> ReviewVectorizer:
        """Returns the vectorizer"""
        return self._vectorizer

    def set_split(self, target_split: str = TRAIN):
        """Selects the splits in the dataset using a column in the dataframe

        Args:
            split (str): one of TRAIN, VALIDATION, TEST
        """
        self._target_split = target_split
        self._target_df, self._target_size = self._splits[target_split]

    def __len__(self):
        return self._target_size

    def __get_item__(self, index: int) -> Dict[str, Any]:
        """The primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        review_vector = self._vectorizer.vectorize(row.text)
        rating_index = self._vectorizer.rating_vocab.lookup_token(row.stars)

        return {"x_data": review_vector, "y_target": rating_index}

    def get_num_batches(self, batch_size: int) -> int:
        """ "Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
