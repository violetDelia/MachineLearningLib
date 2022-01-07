import numpy as np


class AccuracyMixin:
    '''
    计算预测准确度的基类
    '''

    def get_accuracy(self, y_predict, y_test):
        '''
        获取预测准确度

        参数:
            y_predict: 预测的标签值
            y_test: 测试样本的标签
        '''
        correct = 0
        for i in range(y_predict.size):
            if(y_predict[i] == y_test[i]):
                correct += 1
        return correct/y_predict.size
