import numpy as np
from LinearRegression.LinearRegressionBase import LinearRegressionBase


class LassoRegression(LinearRegressionBase):
    '''
    lasso回归模型:
        线性回归添加一范数做惩罚
        f(x) = ||wX-y||^2/m + Lambda_l1|w|/m

    属性:
        w: 各个特征的权重
        Lambda_l1: 一范数正则化系数
        GD_max_steps: 梯度下降最大迭代次数
        GD_step_rate: 梯度下降搜索步长
        GD_epsilon: 梯度下降结果误差许可值
        GD_init_w: 梯度下降的初始值
        power_t: 梯度下降随迭代改变步长的参数
    '''
    Lambda_l1 = None

    def _set_Lambda_l1(self, Lambda_l1):
        self.Lambda_l1 = 1
        if Lambda_l1 is not None:
            if isinstance(Lambda_l1, int) and Lambda_l1 > 0:
                self.Lambda_l1 = float(Lambda_l1)
            if isinstance(Lambda_l1, float) and Lambda_l1 > 0:
                self.Lambda_l1 = Lambda_l1

    def _fit_lasso_SGD(self, X_train, y_train):
        '''
        次梯度下降法求解lasso回归

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        m, n = X_train.shape
        w = self._get_init_w(X_train, y_train)
        sum_w = self._get_init_w(X_train, y_train)
        for i in range(self.GD_max_steps):
            e = X_train.dot(w)-y_train
            subg = (2*X_train.T.dot(e)+self.Lambda_l1*np.sign(w))/m
            w = w - self.GD_step_rate*subg
            sum_w += w
        return sum_w/self.GD_max_steps
