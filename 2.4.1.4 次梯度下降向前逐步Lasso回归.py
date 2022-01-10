import numpy as np
from LinearRegression.LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as skLinearRegression
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso


def generate_sample(m):
    np.random.seed(int(time.time()))
    X = 2*(np.random.rand(m, 1)-0.5)
    y = 10*X**3-2*X**2 + 5*X + 3 + np.random.normal(0, 1, (m, 1))
    return X.reshape(-1, 1), y.reshape(-1, 1)


if __name__ == "__main__":
    X_train, y_train = generate_sample(100)
    X_test, y_test = generate_sample(100)
    X_train_processed = PolynomialFeatures(degree=10).fit_transform(X_train)
    X_test_processed = PolynomialFeatures(degree=10).fit_transform(X_test)

    model = LinearRegression(Lambda_l1=0.01)
    model.train(X_train_processed, y_train, regressionType=model.RegressionType.LassoRegression,
                soulutionType=model.SoulutionType.SGD, processingType=model.ProcessingType.not_process,
                featureSelectionType=model.FeatureSelectType.step_forward)
    y_predict = model.predict(
        X_test_processed, processingType=model.ProcessingType.not_process)

    lasso_model = Lasso(alpha=0.005)
    lasso_model.fit(X_train_processed, y_train)
    linear_y_predict = lasso_model.predict(X_test_processed)

    print("选取的特征: ", model.A)
    print(model.w)
    print("自己写的均方误差: ", model.MSE(y_test, y_predict),
          " 自己写的R2: ", model.R2_score(y_test, y_predict))
    print("sk库lasso回归的均方误差: ", model.MSE(y_test, linear_y_predict),
          " sk库lasso回归的R2: ", lasso_model.score(X_test_processed, y_test))

    model.plot_true_scatter_and_compare_predict_line(
        X_test, y_test, y_predict, linear_y_predict)
    plt.show()
