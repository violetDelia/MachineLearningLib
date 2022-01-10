import numpy as np


class LinearRegressionBase:
    '''
    基础的线性回归
    f(x) = ||wX-y||^2/m

    属性:
        w: 各个特征的权重
        GD_max_steps: 梯度下降最大迭代次数
        GD_step_rate: 梯度下降搜索步长
        GD_epsilon: 梯度下降结果误差许可值
        GD_init_w: 梯度下降的初始值
        power_t: 梯度下降随迭代改变步长的参数
    '''
    w = None
    GD_max_steps = None
    GD_step_rate = None
    GD_epsilon = None
    GD_init_w = None
    power_t = None

    def _set_GD_max_steps(self, GD_max_steps):
        self.GD_max_steps = 3000
        if isinstance(GD_max_steps, int) and GD_max_steps > 0:
            if GD_max_steps > 1000000:
                print("迭代数过大,更新为1000000")
                self.GD_max_steps = 1000000
            else:
                self.GD_max_steps = GD_max_steps
        if isinstance(GD_max_steps, float) and GD_max_steps > 0:
            if GD_max_steps > 1000000:
                self.GD_max_steps = 1000000
                print("迭代数过大,更新为1000000")
            else:
                self.GD_max_steps = int(GD_max_steps)

    def _set_GD_step_rate(self, GD_step_rate):
        self.GD_step_rate = 0.01
        if isinstance(GD_step_rate, float) and GD_step_rate > 0 and GD_step_rate < 1:
            self.GD_step_rate = GD_step_rate

    def _set_GD_epsilon(self, GD_epsilon):
        self.GD_epsilon = 1e-6
        if isinstance(GD_epsilon, float) and GD_epsilon > 0:
            self.GD_epsilon = GD_epsilon

    def _set_GD_init_w(self, GD_init_w):
        self.GD_init_w = None
        if isinstance(GD_init_w, list):
            self.GD_init_w = np.array(GD_init_w).astype(float)
        if isinstance(GD_init_w, np.ndarray):
            self.GD_init_w = GD_init_w.astype(float)

    def _set_power_t(self, power_t):
        self.power_t = 0.25
        if power_t is not None:
            if isinstance(power_t, int) and power_t > 0:
                self.Lambda_l2 = float(power_t)
            if isinstance(power_t, float) and power_t > 0:
                self.Lambda_l2 = power_t

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

    def _get_init_w(self, X_train, y_train):
        '''
        获取初始w点

        参数:
            X_train: 训练特征集
            y_train: 训练结果集
        '''
        m, n = X_train.shape
        my, ny = y_train.shape
        if self.GD_init_w is not None and self.GD_init_w.shape == (n, ny):
            return self.GD_init_w
        else:
            return np.zeros((n, ny))

    def _fit_linear_GD(self, X_train, y_train):
        '''
        梯度下降法求解线性回归

        参数:
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        w = self._get_init_w(X_train, y_train)
        m, n = X_train.shape
        for i in range(self.GD_max_steps):
            e = X_train.dot(w)-y_train
            g = 2*X_train.T.dot(e)/m
            if abs(g.max()) < self.GD_epsilon:
                return w
            w = w - self.GD_step_rate*g
        return w

    def _fit_linear_GD_invscaling(self, X_train, y_train):
        '''
        invscaling梯度下降法求解线性回归

        参数:
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
            power_t: 梯度下降随迭代改变步长的参数
        '''
        w = self._get_init_w(X_train, y_train)
        m, n = X_train.shape
        for i in range(self.GD_max_steps):
            e = X_train.dot(w)-y_train
            g = 2*X_train.T.dot(e)/m
            if abs(g.max()) < self.GD_epsilon:
                return w
            w = w - self.GD_step_rate*g/pow(i+1, self.power_t)
        return w
