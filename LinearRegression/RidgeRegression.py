import numpy as np
from LinearRegression.LinearRegressionBase import LinearRegressionBase


class RidgeRegression(LinearRegressionBase):
    '''
    岭回归模型
        线性回归添加二范数做惩罚
        f(x) = ||wX-y||^2/m + Lambda_l2||w||^2/m

    属性:
        Lambda_l2: 二范数正则化系数
        w: 各个特征的权重
        GD_max_steps: 梯度下降最大迭代次数
        GD_step_rate: 梯度下降搜索步长
        GD_epsilon: 梯度下降结果误差许可值
    '''
    Lambda_l2 = None

    def _set_Lambda_l2(self, Lambda_l2):
        self.Lambda_l2 = 1
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
        g_l2 = np.diag(self.Lambda_l2*np.ones(n))
        return self._l2_normal(X, y, g_l2)

    def _fit_ridge_GD(self, X_train, y_train):
        '''
        梯度下降法求解岭回归

        参数:
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        w = self._get_init_w(X_train, y_train)
        m, n = X_train.shape
        for i in range(self.GD_max_steps):
            e = X_train.dot(w)-y_train
            g = (2*X_train.T.dot(e)+2*self.Lambda_l2*w)/m
            if abs(g.max()) < self.GD_epsilon:
                return w
            w = w - self.GD_step_rate*g
        return w
