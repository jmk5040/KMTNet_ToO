"""Dynamic normalization for model inputs

This script houses a collection of functions designed for image normalization,
making it suitable for preprocessing images before feeding them into models.
The functions cater to various normalization strategies:
    - identity (no changes),
    - min-max normalization (scaling data between 0 and 1),
    - L2 normalization (scaling data to have a L2 norm of 1), and
    - ZScale normalization (specifically visualization for astronomical image).
These normalization techniques are critical for ensuring that model inputs are
on a similar scale, which can significantly impact the training efficiency and
performance of deep learning models.
"""


from   typing import Any

import numpy as np

from   astropy.visualization import LinearStretch, ZScaleInterval
from   albumentations.core.transforms_interface import DualTransform


def identity(x: Any) -> Any:
    """
    Returns the input data unchanged.

    Parameters:
    x (Any): Can be of any data type.

    Returns:
    Any: Returns the input data unchanged.
    """
    return x


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """
    Applies min-max normalization to scale the data between 0 and 1.

    This function is primarily used for preprocessing data that will be used as input to a model.
    It is applied independently to each channel (e.g., ref, new, sub) in the sample images.

    Parameters:
    x (np.ndarray): The NumPy array to be normalized. This could be a single channel of an image.

    Returns:
    np.ndarray: The NumPy array after applying min-max normalization, with values scaled between 0 and 1.
    """
    # If single channel,
    if x.ndim == 2:
        M = x.max()
        m = x.min()
    # If multiple channels of the shape [H, W, C],
    else:
        M = x.max(axis=(0, 1), keepdims=True)
        m = x.min(axis=(0, 1), keepdims=True)
    return (x - m) / (M - m)


def l2_normalization(x: np.ndarray) -> np.ndarray:
    """
    Applies L2 normalization to ensure the L2 norm of the array is 1.

    This function is primarily used for preprocessing data that will be used as input to a model.
    Specifically, this function expects to receive three channels (ref, new, sub) simultaneously,
    meaning the input should be an image with a shape of (H, W, C).
    It is used to normalize the scale of the three channels simultaneously.

    Parameters:
    x (np.ndarray): The NumPy array to be normalized, expected to be an image with shape (H, W, C) where H is height, W is width, and C is the number of channels.

    Returns:
    np.ndarray: The NumPy array after applying L2 normalization across all channels simultaneously.
    """
    # If single channel,
    if x.ndim == 2:
        norm = np.linalg.norm(x)
    # If multiple channels of the shape [H, W, C],
    else:
        norm = np.linalg.norm(x, axis=(0, 1), keepdims=True)
    return x / norm


def zscale_normalization(x: np.ndarray) -> np.ndarray:
    """
    Applies ZScale normalization for astronomical image data visualization.
    This normalization adjusts the image contrast automatically, making celestial bodies like stars and galaxies more visible.
    It uses a combination of LinearStretch and ZScaleInterval transformations from the astropy.visualization module,
    optimizing contrast based on the distribution of pixel brightness.
    For more information, refer to https://iraf.net/forum/viewtopic.php?showtopic=134139

    Parameters:
    x (np.ndarray): The astronomical image data to be normalized, in the form of a NumPy array.

    Returns:
    np.ndarray: The image data after applying ZScale normalization.
    """
    transform = LinearStretch() + ZScaleInterval()
    # If single channel,
    if x.ndim == 2:
        return transform(x)
    # If multiple channels of the shape [H, W, C],
    else:
        _, _, c = x.shape
        x_transformed = np.zeros_like(x)
        for i in range(c):
            x_transformed[:, :, i] = transform(x[:, :, i])
        return x_transformed


class Normalize(DualTransform):
    """
    A class to apply specified normalization techniques to images dynamically.

    This class inherits from DualTransform, allowing it to be integrated into
    pipelines that work with the albumentations library. It provides a flexible
    way to experiment with different image normalization techniques by dynamically
    selecting the method at runtime.

    Parameters:
    method (str): The normalization method to apply.
                  Supported methods are 'identity', 'l2', 'min_max', and 'zscale'.
    always_apply (bool): If True, the transform is always applied. Defaults to True.
    p (float): Probability that the transform will be applied. Defaults to 1.0.

    Attributes:
    normalize_fn (function): The normalization function determined by the `method` parameter.
    """
    def __init__(self, method='identity', always_apply=True, p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        # Build the normalization function based on the specified method.
        self.normalize_fn = self._build(method)

    def _build(self, method):
        """
        Builds the normalization function based on the specified method.

        Parameters:
        method (str): The normalization method to be used.

        Returns:
        function: The corresponding normalization function.

        Raises:
        AssertionError: If an unsupported normalization method is specified.
        """
        assert method in ['identity', 'l2', 'min_max', 'zscale']
        normalize_fn_dict = {
            'identity': identity,
            'l2': l2_normalization,
            'min_max': min_max_normalization,
            'zscale': zscale_normalization
        }
        return normalize_fn_dict[method]

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        Applies the selected normalization technique to an image.

        Parameters:
        img (np.ndarray): The image to be normalized.
        **params: Additional parameters for the normalization function (if any).

        Returns:
        np.ndarray: The normalized image.
        """
        return self.normalize_fn(img)
