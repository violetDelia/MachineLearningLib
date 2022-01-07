import numpy as np
from LinearRegression.LinearRegressionBase import LinearRegressionBase


class RidgeRegression(LinearRegressionBase):
    '''
    岭回归模型
        线性回归添加二范数做惩罚

    属性:
        Lambda_l2: 二范数正则化系数

        w: 各个特征的权重
    '''
    Lambda_l2 = None

    def _set_Lambda_l2(self, Lambda_l2):
        self.Lambda_l2 = 0.001
        if Lambda_l2 is not None:
            if isinstance(Lambda_l2, int) and Lambda_l2 > 0:
                self.Lambda_l2 = float(Lambda_l2)
            if isinstance(Lambda_l2, float) and Lambda_l2 > 0:
                self.Lambda_l2 = Lambda_l2

    def _l2_normal(self, X, y, g_l2):
        '''
        正则化求解带二范数的回归

        参数:
            X: 处理好的训练特征集
            y: 处理好的训练结果集
            g_l2: 二范数梯度项
        '''
        return np.linalg.inv(X.T.dot(X) + g_l2).dot(X.T).dot(y)

    def _fit_ridge_normal(self, X, y):
        '''
        使用正则化方程直接求得岭回归的最优解

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        m, n = X.shape
        g_l2 = m * np.diag(self.Lambda_l2*np.ones(n))
        return self._l2_normal(X, y, g_l2)
