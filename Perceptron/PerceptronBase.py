import numpy as np
from Perceptron.PerceptronUtility import PerceptronUtility


class PerceptronBase(PerceptronUtility):
    '''
    最基础的感知机

    属性:
        w: 各个特征的权重
            是一个(n,1)的数组.

        b: 偏移量

        start_w: 搜索w的初始点
            如果设置了,那么会从start_w开始搜索解的w,否则,会从零向量开始搜索.

        start_b: 搜索b的初始点
            如果设置了,那么会从start_b开始搜索解的b,否则,会从0开始搜索.

        learning_rate: 学习率
                迭代搜索符合假设的超平面时的步长
    '''
    w_start = None
    w = None
    b_start = None
    b = None
    max_steps = None
    learning_rate = None

    def _set_w_start(self, w_start):
        self.w_start = None
        if isinstance(w_start, list):
            self.w_start = np.array(w_start).reshape(1, -1).astype(float)
        if isinstance(w_start, np.ndarray):
            self.w_start = w_start.reshape(1, -1).astype(float)

    def _set_b_start(self, b_start):
        self.b_start = 0
        if isinstance(b_start, int):
            self.b_start = float(b_start)
        if isinstance(b_start, float):
            self.b_start = b_start

    def _set_max_steps(self, max_steps):
        self.max_steps = 3000
        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        if isinstance(max_steps, float) and max_steps > 1:
            self.max_steps = int(max_steps)

    def _set_learning_rate(self, learning_rate):
        self.learning_rate = 1
        if isinstance(learning_rate, int) and learning_rate > 0:
            self.learning_rate = float(learning_rate)
        if isinstance(learning_rate, float) and learning_rate > 0:
            self.learning_rate = float(learning_rate)

    def _get_initial_points(self, n):
        '''
        获取初始点.

        参数:
            n: 特征个数

        返回:
            w_start: 搜索w的初始点
            b_start: 搜索b的初始点
        '''
        if self.w_start is not None and self.w_start.shape == (1, n):
            w_start = self.w_start
        else:
            w_start = np.zeros((1, n))
        b_start = self.b_start
        return w_start, b_start

    def _fit_normal(self, X_train, y_train):
        '''
        基本感知机的搜索算法

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
        '''

        m, n = X_train.shape
        w_current, b_current = self._get_initial_points(n)
        is_find = False
        steps = 0
        while not is_find and steps < self.max_steps:
            is_find = True
            for i in range(m):
                Xi = X_train[i].reshape(1, -1)
                if y_train[i] * (w_current.dot(Xi.T)+b_current) <= 0:
                    w_current += Xi*y_train[i]*self.learning_rate
                    b_current += y_train[i]*self.learning_rate
                    is_find = False
            steps += 1
        return w_current, b_current
