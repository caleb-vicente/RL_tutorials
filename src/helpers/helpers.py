import numpy as np


def moving_average(data, window_size):
    # Pad the data at the beginning to handle edge cases
    padded_data = np.pad(data, (window_size - 1, 0), mode='constant')
    # Calculate the moving average using convolution
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_data, weights, mode='valid')
    return moving_avg
