import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = 5*X + np.random.normal(0, 0.5, (m, 1))+3
    return X.reshape(-1, 1), y.reshape(-1, 1)


if __name__ == "__main__":
    X_train, y_train = generate_sample(100)
    X_test, y_test = generate_sample(100)

    model = LinearRegression()
    model.train(X_train, y_train, regressionType=model.RegressionType.LinearRegression,
                soulutionType=model.SoulutionType.GD_invscaling, processingType=model.ProcessingType.normal)
    y_predict = model.predict(X_test)

    linear_model = Pipeline([
        ("lin_reg", skLinearRegression())
    ])
    linear_model.fit(X_train, y_train)
    linear_y_predict = linear_model.predict(X_test)

    SGD_model = SGDRegressor()
    SGD_model.fit(X_train, y_train.ravel())
    sky_predict = SGD_model.predict(X_test)

    print("自己写的均方误差: ", model.MSE(y_test, y_predict),
          " 自己写的R2: ", model.R2_score(y_test, y_predict))
    print("sk库线性回归的均方误差: ", model.MSE(y_test, linear_y_predict),
          " sk库线性回归的R2: ", linear_model.score(X_test, y_test))
    print("sk库随机梯度的均方误差: ", model.MSE(y_test, sky_predict),
          " sk库随机梯度的R2: ", SGD_model.score(X_test, y_test))

    model.plot_true_scatter_and_compare_predict_line(
        X_test, y_test, y_predict, linear_y_predict)
    plt.show()
