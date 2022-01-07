import numpy as np
from Perceptron.Perceptron import Perceptron
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.linear_model import Perceptron as skPerceptron


def load_data(remove_column):
    filepath = "datasets\Iris\Iris.csv"
    data = pd.read_csv(filepath)
    '''
    去掉没用的列
    '''
    data.drop(remove_column, axis=1, inplace=True)
    return data


def get_train_and_test_data(data, lable, test_size, random_state):
    return sklearn.model_selection.train_test_split(data, lable, test_size=test_size, random_state=random_state)


def set_lable(data):
    lable_group = data.groupby([data.columns[4]])
    lable_list = []
    for name, info in lable_group:
        lable_list.append(name)
    data["lable"] = data[data.columns[4]].replace(
        {lable_list[0]: -1, lable_list[1]: 1, lable_list[2]: 1})


if __name__ == "__main__":
    remove_column = ["Id"]
    data = load_data(remove_column)
    set_lable(data)
    train_x, test_x, train_y, test_y = get_train_and_test_data(
        data[[data.columns[0], data.columns[1]]], data["lable"], 0.5, 8)

    data_t = {"x": test_x.values[:, 0],
              "y": test_x.values[:, 1], "lable": test_y.values}
    data_t = pd.DataFrame(data_t)

    model = Perceptron(learning_rate=10,w_start=[1,1], b_start=None,
                       max_steps=1000, nonexact_accuracy_rate=1, nonexact_steps=50)
    model.train(train_x.values, train_y.values,
                model.SoulutionType.normal_nonexact)
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
