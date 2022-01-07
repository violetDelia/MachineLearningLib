import numpy as np
from ModelEvaluation.MSEMixin import MSEMixin


class R2Base(MSEMixin):
    '''
    决定系数的基类
    '''

    def R2_score(self, true, predict):
        '''
        计算决定系数

        参数：
            true: 真实的数据集
            predict: 预测的数据集
        '''

        true = true.reshape(-1, 1)
        predict = predict.reshape(-1, 1)
        average = np.average(true, axis=0)
        distance = true-average
        return 1 - self.MSE_sum(true, predict)/distance.T.dot(distance)[0, 0]
