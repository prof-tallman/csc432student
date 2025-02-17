import os
import pandas as pd
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from PIL import Image

if __name__ == '__main__':

    # Load some data from the MNIST database, we only want 100 samples
    mnist_local_file = 'mnist.csv'
    if os.path.exists(mnist_local_file) and os.path.isfile(mnist_local_file):
        print("\nLoading MNIST training data from local file")
        data = np.loadtxt(mnist_local_file, delimiter=",")
        X = data[:,1:].astype(np.uint8)
        y = data[:,0].astype(np.uint8)
    else:
        print("\nDownloading and caching MNIST training data from internet")
        mnist = fetch_openml('mnist_784', parser='auto', 
                             version=1, as_frame=False)
        X = mnist.data.astype(np.uint8)[:100]
        y = mnist.target.astype(np.uint8)[:100]
        np.savetxt(mnist_local_file, np.column_stack((y, X)), 
                   delimiter=',', fmt='%d')

    # Get a sigle digit from MNIST
    some_digit = X[0]
    print(f"Original Size: {some_digit.shape}")
    print(f"\n{some_digit}\n")

    # We need a 2-D array to display this image
    some_digit = some_digit.reshape(28, 28)
    print(f"Resized Size:  {some_digit.shape}")
    print(f"\n{some_digit}\n")
    plt.imshow(some_digit, cmap=cmap.gray)
    plt.axis("off")
    plt.show()

    # Convert to Pillow image and save
    pimage = Image.fromarray(some_digit)
    pimage.show()
    print("Done with MNIST")
