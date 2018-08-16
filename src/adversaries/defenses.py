import numpy as np
import torch


def bit_squeeze(x, bits=7):
    """
    Implementation of colour bit squeezing.

    :param x: Original image
    :type x: NumPy Array
    :param bits: The number of bits to reduce the image too
    :type bits: int
    :return: Squeezed image
    :rtype: NumPy Array
    """
    # 2^i -1
    bytes = (2**bits)-1

    # Multiply and round
    intermediate = int(np.round(x*bytes))

    # Rescale
    x_squeezed = intermediate/bytes
    return x_squeezed


def bit_squeeze_torch(x, bits=7):
    """
    Perform bit squeezing on a torch array which will be of shape [n, 1, w, h] for a batch of n images

    :param x: Tensor of images
    :type x: Tensor
    :param bits: Bits for which images should be scaled to
    :type bits: 7
    :return: Tensor of squeezed bits
    """
    bytes = (2**bits)-1

    # Multiply and round
    intermediate = torch.round(torch.mul(x, bytes))

    # Rescale
    x_squeezed = torch.div(intermediate, bytes)
    return x_squeezed