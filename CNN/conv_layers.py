import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding_mode="same"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode

        # Initialize the filters (weights) for the convolutional layer
        if self.padding_mode == "same":
            padding_mode = (self.kernel_size - 1) // 2
        elif self.padding_mode == "valid":
            padding_mode = 0
        elif isinstance(self.padding_mode, int):
            padding_mode = self.padding_mode
        else:
            raise ValueError("Invalid padding_mode. Use 'same', 'valid', or an integer.")
