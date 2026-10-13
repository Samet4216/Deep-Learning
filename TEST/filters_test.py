import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from CNN.filters import convolve2D_single_chanel, SOBEL_X, SOBEL_Y, LAPLACIAN

if __name__ == "__main__":
    image_test = np.zeros((6, 6), dtype=np.float32)
    image_test[:, 3] = 1 #3. y label full =1
    print(image_test)
    feature_map = convolve2D_single_chanel(image_test, SOBEL_X, stride=1, padding=1) 
    print(feature_map)