"""Trainer class for the Language Model."""

import torch


class Trainer:
    """Trainer module."""

    def __init__(
        self, batch_size: int, block_size: int, train_test_split: float, data: list[int]
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        transformed_data = torch.tensor(data, dtype=torch.long)
        self.threshold = int(train_test_split * len(transformed_data))
        self.train_set = transformed_data[: self.threshold]
        self.validation_set = transformed_data[self.threshold :]

    def get_train_data(self):
        """Get the Training split."""
        return self.train_set

    def get_validation_data(self):
        """Get the Validation split."""
        return self.validation_set

    def get_batch_of_train_or_test_split(self, split):
        """
        Return batch of Train or Validation split.
        The batch holds the input and the target list of a block size.
        Eg. For a block size of 5
        And input: [5, 1, 4, 18, 12, 10, 30]
        Return -> [5, 1, 4, 18, 12, 10], [1, 4, 18, 12, 10, 30]
        The Target array holds the target index (to the right) of the input.
        """
        data = self.train_set if split == "train" else self.validation_set
        # For a 'batch_size' of 5
        # Return 5 random indices in the range of [0, len(data) - block_size]
        # We take the 'block_size' offset coz we need to generate the input and target
        # of length block_size. If it goes more than that then we can fall into an Index Error.
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])

        return x, y
