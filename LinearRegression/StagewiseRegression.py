import numpy as np


class StagewiseRegression:
    '''
    分段回归

    参数:
        stagewize_learning_rate: 分段回归算法中的学习率
        stagewize_max_steps: 分段回归迭代次数
    '''
    stagewize_learning_rate = None
    stagewize_max_steps = None

    def _set_stagewize_learning_rate(self, stagewize_learning_rate):
        self.stagewize_learning_rate = 0.01
        if isinstance(stagewize_learning_rate, float) and stagewize_learning_rate > 0:
            self.stagewize_learning_rate = stagewize_learning_rate
        if isinstance(stagewize_learning_rate, int) and stagewize_learning_rate > 0:
            self.stagewize_learning_rate = stagewize_learning_rate

    def _set_stagewize_max_steps(self, stagewize_max_steps):
        self.stagewize_max_steps = 10000
        if isinstance(stagewize_max_steps, int) and stagewize_max_steps > 0:
            if stagewize_max_steps > 1000000:
                self.stagewize_max_steps = 1000000
            else:
                self.stagewize_max_steps = stagewize_max_steps
        if isinstance(stagewize_max_steps, float) and stagewize_max_steps > 0:
            if stagewize_max_steps > 1000000:
                self.stagewize_max_steps = 1000000
            else:
                self.stagewize_max_steps = int(stagewize_max_steps)

    def _fit_stagewize(self, X, y):
        '''
        分段回归寻找w

        参数:
            X: 特征集
            y: 结果集
        '''
        m, n = X.shape
        w = np.zeros(n)
        # 计算每列的二范数
        norms = np.linalg.norm(X, 2, axis=0).reshape(-1, 1)
        step = 0
        # 误差向量
        distances = y
        while step < self.stagewize_max_steps:
            # 相关系数
            corr = X.T.dot(distances)/norms
            # 相关性最高的那组特征
            j_max = np.argmax(abs(corr))
            if corr[j_max]< self.stagewize_learning_rate:
                break
            if w[j_max] == 0:
                w[j_max] = self.stagewize_learning_rate
                distances = distances - self.stagewize_learning_rate * X[:, j_max].reshape(-1, 1)
            else:
                delta = self.stagewize_learning_rate * np.sign(corr[j_max])
                w[j_max] = w[j_max] + delta
                distances = distances - delta * X[:, j_max].reshape(-1, 1)
            step += 1
        return w
