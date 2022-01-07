import numpy as np


class LinearRegressionBase:
    '''
    基础的线性回归

    属性:
        w: 各个特征的权重
    '''
    w = None

    def _normal(self, X, y):
        '''
        用正规方程求得w

        参数:
            X: 处理好的训练特征集
            y: 处理好的训练结果集
        '''
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def _fit_linear_normal(self, X_train, y_train):
        '''
        使用正则化方程直接求得最优解

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        return self._normal(X_train, y_train)

    def _fit_linear_GD(self, X_train, y_train, GD_max_steps, GD_step_rate, GD_epsilon):
        '''
        梯度下降法求解线性回归

        参数:
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        m, n = X_train.shape
        w = np.zeros((n, 1))
        for i in range(GD_max_steps):
            e = X_train.dot(w)-y_train
            g = 2*X_train.T.dot(e) / m
            if abs(g.max()) < GD_epsilon:
                return w
            w = w - GD_step_rate*g
        return w
