import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.datasets import fetch_olivetti_faces





if __name__ == "__main__":
    data = fetch_olivetti_faces()
    images = data.images
    plt.imshow(images[0])
    plt.show()

    data = images.reshape((len(data.images,-1)))
    n_pixels = data.shape[1]
    X = data[:,:(n_pixels+1)//2]
    y = data[:,n_pixels//2:]

    
