import math

from torch import nn


def closest_larger_multiple_of_minimum_size(size, minimum_size):
    return int(math.ceil(size / minimum_size) * minimum_size)

class SizeAdapter(object):
    """Converts size of input to standard size.
    Practical deep network works only with input images
    which height and width are multiples of a minimum size.
    This class allows to pass to the network images of arbitrary
    size, by padding the input to the closest multiple
    and unpadding the network's output to the original size.
    """

    def __init__(self, minimum_size=64):
        self._minimum_size = minimum_size
        self._pixels_pad_to_width = None
        self._pixels_pad_to_height = None

    def _closest_larger_multiple_of_minimum_size(self, size):
        return closest_larger_multiple_of_minimum_size(size, self._minimum_size)

    def pad(self, network_input):
        """Returns "network_input" paded with zeros to the "standard" size.
        The "standard" size correspond to the height and width that
        are closest multiples of "minimum_size". The method pads
        height and width  and and saves padded values. These
        values are then used by "unpad_output" method.
        """
        height, width = network_input.size()[-2:]
        self._pixels_pad_to_height = int((self._closest_larger_multiple_of_minimum_size(height) - height)/2)
        self._pixels_pad_to_width = int((self._closest_larger_multiple_of_minimum_size(width) - width)/2)
        return nn.ReflectionPad2d((self._pixels_pad_to_width, self._pixels_pad_to_width, self._pixels_pad_to_height, self._pixels_pad_to_height))(network_input)

    def unpad(self, network_output):
        """Returns "network_output" cropped to the original size.
        The cropping is performed using values save by the "pad_input"
        method.
        """
        if self._pixels_pad_to_height==0 and self._pixels_pad_to_width==0 :
            return network_output

        elif self._pixels_pad_to_height==0:
            return network_output[..., :,
                   self._pixels_pad_to_width:-self._pixels_pad_to_width]
        elif self._pixels_pad_to_width==0:
            return network_output[..., self._pixels_pad_to_height:-self._pixels_pad_to_height,:
                   ]
        else:
            return network_output[..., self._pixels_pad_to_height:-self._pixels_pad_to_height, self._pixels_pad_to_width:-self._pixels_pad_to_width
                   ]

