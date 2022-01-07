import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = 10*X**3-2*X**2 + 5*X + 3 + np.random.normal(0, 0.5, (m, 1))
    return X.reshape(-1, 1), y.reshape(-1, 1)


if __name__ == "__main__":
    X_train, y_train = generate_sample(100)
    X_test, y_test = generate_sample(100)
    weights = []
    for i in range(100):
        weights.append(2)
    weights = np.array(weights)
    X_train_processed = PolynomialFeatures(degree=10).fit_transform(X_train)
    X_test_processed = PolynomialFeatures(degree=10).fit_transform(X_test)

    model = LinearRegression()
    model.train(X_train_processed, y_train, regressionType=model.RegressionType.RidgeRegression,
                soulutionType=model.SoulutionType.normal, processingType=model.ProcessingType.not_process, weights=weights)
    y_predict = model.predict(
        X_test_processed, processingType=model.ProcessingType.not_process, weights=weights)

    noweight_model = LinearRegression()
    noweight_model.train(X_train_processed, y_train, regressionType=model.RegressionType.RidgeRegression,
                         soulutionType=model.SoulutionType.normal, processingType=model.ProcessingType.not_process)
    noweight_y_predict = noweight_model.predict(
        X_test_processed, processingType=noweight_model.ProcessingType.not_process)

    skmodel = Ridge()
    skmodel.fit(X_train_processed, y_train, sample_weight=weights)
    sky_predict = skmodel.predict(X_test_processed)

    print("无权重的均方误差: ", noweight_model.MSE(y_test, noweight_y_predict),
          " 无权重的R2: ", noweight_model.R2_score(y_test, noweight_y_predict))
    print("自己写的均方误差: ", model.MSE(y_test, y_predict),
          " 自己写的R2: ", model.R2_score(y_test, y_predict))
    print("sk库线性回归的均方误差: ", model.MSE(y_test, sky_predict),
          " sk库线性回归的R2: ", skmodel.score(X_test_processed, y_test))

    model.plot_true_scatter_and_compare_predict_line(
        X_test, y_test, y_predict, sky_predict)
    plt.show()
