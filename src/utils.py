import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_grayscale_uint8(image_tensor):
    """
    Shifts and rescales tensor values to the range [0,255] and converts the tensor to a uint8. This is useful for displaying the tensor as an image. In particular, PIL expects grayscale images in this format.
    """

    # shift all values to make the minimum value 0
    im = image_tensor - image_tensor.min()

    # devide by (max/255) to rescale the maximum to 255
    im /= im.max() / 255

    # convert to uint8
    im = im.to(torch.uint8)

    return im


def distribute_uniformly(n, n_bins):
    """
    Distribute n items across n_bins bins as uniformly as possible. If n is not divisible by n_bins, some bins get one extra.

    Returns a mapping from the index of the bin to the number of items assigned to that bin. E.g. distribute_uniformly(n, n_bins)[i] returns the number of items in bin i.
    """
    quotient = n // n_bins
    remainder = n % n_bins

    # each bin gets at least the quotient number of items
    bin_size = [quotient for _ in range(n_bins)]

    # distribute the remaining items across equally many bins
    for i in range(remainder):
        bin_size[i] += 1

    return bin_size
