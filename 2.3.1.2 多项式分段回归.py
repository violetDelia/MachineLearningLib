import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = 10*X**3-2*X**2 + 5*X + 3 + np.random.normal(0, 0.5, (m, 1))
    return X.reshape(-1, 1), y.reshape(-1, 1)


if __name__ == "__main__":
    X_train, y_train = generate_sample(100)
    X_test, y_test = generate_sample(100)

    model = LinearRegression(
        stagewize_learning_rate=0.01, stagewize_max_steps=3000)
    model.train(X_train, y_train, regressionType=model.RegressionType.StagewiseRegression,
                soulutionType=model.SoulutionType.normal, processingType=model.ProcessingType.multinomial,
                processing_feature_degree=10)
    y_predict = model.predict(
        X_test, processingType=model.ProcessingType.multinomial, processing_feature_degree=10)

    linear_model = Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("lin_reg", skLinearRegression())
    ])

    linear_model.fit(X_train, y_train)
    linear_y_predict = linear_model.predict(X_test)

    print("自己写的均方误差: ", model.MSE(y_test, y_predict),
          " 自己写的R2: ", model.R2_score(y_test, y_predict))
    print("sk库线性回归的均方误差: ", model.MSE(y_test, linear_y_predict),
          " sk库线性回归的R2: ", linear_model.score(X_test, y_test))

    model.plot_true_scatter_and_compare_predict_line(
        X_test, y_test, y_predict, linear_y_predict)
    plt.show()
