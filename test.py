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
    y1 = 5*X + np.random.normal(0, 0.5, (m, 1))+3
    y2 = 18*X + np.random.normal(0, 0.5, (m, 1))+8
    y = np.concatenate((y1, y2), axis=0)
    return X.reshape(-1, 1), y.reshape(m, 2)


if __name__ == "__main__":
    X_train, y_train = generate_sample(100)
    X_test, y_test = generate_sample(100)
    weights = []
    for i in range(100):
        weights.append(2)
    weights = np.array(weights)

    model = LinearRegression()
    model.train(X_train, y_train, regressionType=model.RegressionType.LinearRegression,
                soulutionType=model.SoulutionType.normal, processingType=model.ProcessingType.multinomial,
                processing_feature_degree=10, weights=weights)
    y_predict = model.predict(
        X_test, processingType=model.ProcessingType.multinomial, processing_feature_degree=10, weights=weights)

    noweight_model = LinearRegression()
    noweight_model.train(X_train, y_train, regressionType=noweight_model.RegressionType.LinearRegression,
                         soulutionType=noweight_model.SoulutionType.normal,
                         processingType=noweight_model.ProcessingType.multinomial, processing_feature_degree=10)
    noweight_y_predict = noweight_model.predict(
        X_test, processingType=noweight_model.ProcessingType.multinomial, processing_feature_degree=10)


    X_train_processed = PolynomialFeatures(degree=10).fit_transform(X_train)
    skmodel =  skLinearRegression()
    skmodel.fit(X_train_processed, y_train)
    X_test_processed = PolynomialFeatures(degree=10).fit_transform(X_test)
    sky_predict = skmodel.predict(X_test_processed)


    print("无权重的均方误差: ", noweight_model.MSE(y_test, noweight_y_predict),
          " 无权重的R2: ", noweight_model.R2_score(y_test, noweight_y_predict))
    print("自己写的均方误差: ", model.MSE(y_test, y_predict),
          " 自己写的R2: ", model.R2_score(y_test, y_predict))
    print("sk库线性回归的均方误差: ", model.MSE(y_test, sky_predict),
          " sk库线性回归的R2: ", skmodel.score(X_test_processed, y_test))

    

    
