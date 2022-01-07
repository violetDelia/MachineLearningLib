from Perceptron.Perceptron import Perceptron
from sklearn.linear_model import Perceptron as skPerceptron
import numpy as np
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection
import seaborn as sns


def get_data():
    X, y = make_blobs(n_samples=200, n_features=2,
                      cluster_std=0.6, random_state=0)
    data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "lable": y})
    data["lable"] = data["lable"].replace({0: 1, 2: 1, 1: -1})
    return data


if __name__ == "__main__":
    data = get_data()
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
        data[["x", "y"]], data["lable"], test_size=0.5)
   
    data_t = {"x": test_x.values[:, 0],
              "y": test_x.values[:, 1], "lable": test_y.values}
    data_t = pd.DataFrame(data_t)

    model = Perceptron(learning_rate= 5,w_start=[1,1],b_start=None)
    model.train(train_x.values, train_y.values)
    print(model.w.shape)
    predict_y = model.predict(test_x.values)

    skmodel = skPerceptron()
    skmodel.fit(train_x.values, train_y.values)
    skpredict_y = model.predict(test_x.values)

    print("自己写的准确率: ",model.get_accuracy(predict_y, test_y.values))
    print("skle库的准确率: ",skmodel.score(test_x.values, test_y.values))
    print(model.w)
    print(skmodel.coef_)
    print(model.b)
    print(skmodel.intercept_)
    
    model.plot_2D_scatter_and_line(
        data_t, "x", "y", "lable", model.w, model.b, "red")
    model.plot_2D_scatter_and_line(
        data_t, "x", "y", "lable", skmodel.coef_.reshape(-1, 1), skmodel.intercept_)
    plt.show()
