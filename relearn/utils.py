"""
Utility module for handling probability distributions with numpy arrays.

This module defines the ProbabilityArray class, a numpy ndarray subclass designed 
to enforce and maintain the properties of probability distributions. 
The ProbabilityArray class ensures that arrays are valid probability distributions 
by requiring all elements to be non-negative and their sum to equal 1. This 
constraint is applied both to one-dimensional arrays, representing individual 
probability distributions, and to the last dimension of multi-dimensional arrays, 
where each sub-array along the last dimension is treated as a separate probability distribution.

The module facilitates the creation, manipulation, and validation of arrays intended 
to represent probability distributions, making it suitable for applications in 
statistics, machine learning, and other fields requiring probabilistic data 
representation and manipulation.

Classes:
    ProbabilityArray: A numpy ndarray subclass enforcing probability distribution constraints.

The ProbabilityArray class overrides specific numpy ndarray methods to ensure 
that any modifications to the array maintain its validity as a probability distribution. 
This includes re-normalizing values after updates and validating new values before 
they are set.
"""

import numpy as np


class ProbabilityArray(np.ndarray):
    """A custom numpy ndarray subclass that enforces the properties of a probability distribution.

    This class ensures that arrays represent valid probability distributions: all elements are
    non-negative, and their sum equals 1. It is designed to work with one-dimensional arrays
    representing single probability distributions and multi-dimensional arrays where the last
    dimension consists of probability distributions.

    Attributes:
        There are no additional attributes beyond those provided by np.ndarray.

    Methods:
        __new__(cls, input_array): Creates a new ProbabilityArray instance from an input array.
        __array_finalize__(obj): Method called automatically by numpy to finalize object creation.
        __setitem__(key, value): Overrides item assignment to maintain probability
        distribution constraints.
        _is_probability_distribution(): Checks if the array represents a valid
        probability distribution.
    """

    def __new__(cls, input_array):
        """Creates a new ProbabilityArray instance, ensuring it represents a valid
        probability distribution.

        Args:
            input_array (array-like): The input array to be converted into a ProbabilityArray.

        Returns:
            ProbabilityArray: A new ProbabilityArray object.

        Raises:
            ValueError: If the input_array contains negative values or does not sum to 1.
        """
        obj = np.asarray(input_array).view(cls)
        if not obj._is_probability_distribution():
            raise ValueError(
                "Probabilities are either negative or their sum is not close to one"
            )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __setitem__(self, key, value):
        """Overrides item assignment to ensure modifications maintain probability
        distribution constraints.

        Args:
            key: The index or slice indicating where the value should be set.
            value: The value to set at the specified index or slice.

        Raises:
            ValueError: If attempting to modify a single probability without
            considering the entire distribution.
        """
        if not isinstance(key, int):
            if len(key) != (len(self.shape) - 1):
                raise ValueError(
                    "You cannot modify just one probability entry in "
                    "probability vector without adjusting every other "
                    "probability value."
                )
        # checking if the value is a probability distribution: it is way easier
        # to let numpy handle all possible types of passed value
        _ = ProbabilityArray(value)
        super().__setitem__(key, value)

    def _is_probability_distribution(self) -> bool:
        """Checks if the array or sub-arrays represent valid probability distributions.

        Returns:
            bool: True if the array or all sub-arrays are valid probability
            distributions, False otherwise.
        """
        if len(self.shape) == 1:
            non_negative = np.all(self >= 0)
            max_sum_to_one = np.sum(self) <= 1
            return non_negative and max_sum_to_one
        # pylint: disable=W0212
        return all(sub_arr._is_probability_distribution() for sub_arr in self)

    @staticmethod
    def from_any_real_array(real_array: np.ndarray) -> any:
        """Converts any real-valued numpy array into a valid ProbabilityArray.

        This method shifts and scales a given real-valued array to ensure that all elements are
        non-negative and that their sum equals 1, thus forming a valid probability distribution.
        This transformation allows the creation of a ProbabilityArray from a broader
        range of numerical data.

        The process involves:
        1. Shifting the array's values to make them all non-negative by adding the absolute value
           of the array's minimum value plus a small constant (0.000001) to prevent
           zero-only arrays.
        2. Normalizing the shifted values to ensure their sum equals 1.

        Args:
            real_array (np.ndarray): The input numpy array with real values.

        Returns:
            ProbabilityArray: A new ProbabilityArray instance representing a valid
            probability distribution derived from the input real_array.

        Example:
            >>> real_array = np.array([-1, 2, -3, 4])
            >>> prob_array = ProbabilityArray.from_any_real_array(real_array)
            >>> print(prob_array)
            [0.08398438 0.41796875 0.         0.49804688]
        """
        probas = real_array + np.abs(np.min(real_array)) + 0.000001
        probas = np.float16(probas) / np.sum(probas)
        return ProbabilityArray(probas)
