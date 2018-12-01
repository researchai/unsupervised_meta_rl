import numpy as np


class Dataset:
    """Methods to retrieve data from a dataset within a numpy file.

    This class implements the fluent interface to retrieve a subset of data
    from the entire dataset. For example, if our dataset contains a dictionary,
    we can call the following function to get the array for one of the
    corresponding keys:
        dataset.fromKey("key_1")
    If we only care about the entries with index 3 and 4 within the previous
    subset, we can call:
        dataset.fromIndex(3, 4)
    We can also use names if they're set within the dtype of the arrays:
        dataset.fromIndex("name_1", "name_2")
    To reduce the length of each entry we can call:
        dataset.withLength(5, axis=1)
    And finally we call get to obtain the final array with:
        dataset.get()
    This can be done in a single line as well:
        dataset.fromKey("key_1").fromIndex(3, 4).withLength(5, axis=1).get()
    """

    def __init__(self, file_path):
        """Start the dataset from a numpy file

        Parameters:
            file_path: path where the numpy file containing the dataset is
            located.
        """
        self.data = np.load(file_path)
        self.processed_data = self.data
        self.returned_data = self.data

    def fromKey(self, key):
        """Get the array from a dictionary within the main numpy array.

        Parameters:
            key: key for value in the dictionary.

        Returns:
            The instance where this method was called.
        """
        assert hasattr(
            self.processed_data[()],
            "keys"), ("The numpy array does not contain a dictionary")
        assert key in self.processed_data[()].keys()
        self.processed_data = self.processed_data[()][key]
        return self

    def fromIndex(self, *indices):
        """Select specific entries within the main axis of the numpy array.

        Parameters:
            indices: single or multiple indices of entries in the numpy array.

        Returns:
            The instance where this method was called.
        """
        indices = [*indices]
        assert indices
        if isinstance(indices[0], int):
            for index in indices:
                assert 0 <= index and index < self.processed_data.shape[0]
        elif isinstance(indices[0], str):
            for index in indices:
                assert index in self.processed_data.dtype.names, (
                    "There's no data with name " + index + ". Try with: " +
                    " ".join(self.processed_data.dtype.names))
        else:
            raise NotImplementedError

        if len(indices) is 1:
            self.processed_data = self.processed_data[indices[0]]
            return self
        else:
            self.processed_data = self.processed_data[[*indices]]
            return self

    def withLength(self, length, start_index=0, axis=0):
        """Set the length of an axis within the sub-array.

        Parameters:
            length: number of items to select from the specified axis.
            start_index: index from where items start to be selected.
            axis: axis in the sub-array where items are selected.
        """
        assert (isinstance(start_index, int) and start_index >= 0)
        data_shape = self.processed_data.shape
        assert (isinstance(axis, int) and (0 <= axis < len(data_shape)))
        assert (isinstance(length, int)
                and length <= (data_shape[axis] - start_index))
        indices = list(range(start_index, start_index + length))
        self.processed_data = np.take(self.processed_data, indices, axis)
        return self

    def get(self):
        """Obtain the sub-array filtered by the methods in this class.

        Once get is called, a new sub-array can be created.
        """
        self.returned_data = self.processed_data
        self.processed_data = self.data
        return self.returned_data
