import numpy as np

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

LAPLACIAN = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

def convolve2D_single_chanel(image, kernel, stride=1, padding=0): #stride=> step size
    image_H, image_W = image.shape
    karnel_H, karnel_W = kernel.shape
    padded_image = image

    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        #(padding, padding) => padding for up and bottom, (padding, padding) => padding for left and right
        #mode='constant' => fill the padding with constant values, constant_values=0 => fill with 0
    else:
        padded_image = image

    padded_image_H, padded_image_W = padded_image.shape
    H_out = (padded_image_H - karnel_H) // stride + 1 
    W_out = (padded_image_W - karnel_W) // stride + 1

    feature_map = np.zeros((H_out, W_out), dtype=np.float32) 

    for i in range(H_out):
        for j in range(W_out):
            H_start = i * stride
            H_end = H_start + karnel_H
            W_start = j * stride
            W_end = W_start + karnel_W

            patch = padded_image[H_start:H_end, W_start:W_end]
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map
