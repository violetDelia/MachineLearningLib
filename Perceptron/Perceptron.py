import numpy as np

from enum import Enum
from Perceptron.PerceptronNonexact import PerceptronNonexact
from ModelEvaluation.ModelEvaluation import ModelEvaluation
from DataVisualization.DateVisualization import DataVisualization


class Perceptron(PerceptronNonexact, ModelEvaluation, DataVisualization):
    ''' 
    感知机:
        寻找超平面 wx+b = 0 可以将样本分开.

    属性:
        w: 各个特征的权重
            是一个(n,1)的数组.

        b: 偏移量

        start_w: 搜索w的初始点
            如果设置了,那么会从start_w开始搜索解的w,否则,会从零向量开始搜索.

        start_b: 搜索b的初始点
            如果设置了,那么会从start_b开始搜索解的b,否则,会从0开始搜索.

        max_steps: 迭代的最大次数
            如果迭代超过max_steps还没有找到解的话,退出搜索,不更新w,b值.

        nonexact_accuracy_rate : 非精确求解时需要满足的合格率
            如果采用非精确求解,那么每迭代NonExactSolution_step次后会检验一次合格率,如果预测满足该合格率,那么认为该解是可行解

        nonexact_steps: 非精确求解检验合格率的迭代参数
    '''

    class SoulutionType(Enum):
        '''
        normal: 一般求解方法
        normal_nonexact: 非精确求解方法
        '''
        normal = 0
        '''
        一般求解方法
        '''
        normal_nonexact = 1
        '''
        非精确求解方法
        '''

    def __init__(self,learning_rate=1, w_start=None, b_start=None, max_steps=3000,
                 nonexact_accuracy_rate=0.95, nonexact_steps=50):
        self._set_learning_rate(learning_rate)
        self._set_w_start(w_start)
        self._set_b_start(b_start)
        self._set_max_steps(max_steps)
        self._set_nonexact_accuracy_rate(nonexact_accuracy_rate)
        self._set_nonexact_steps(nonexact_steps)

    def train(self, X_train, y_train, type=SoulutionType.normal):
        '''
        训练模型

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            type: 求解类型
        '''
        if type == self.SoulutionType.normal:
            self.w, self.b = self._fit_normal(X_train, y_train)
        if type == self.SoulutionType.normal_nonexact:
            self.w, self.b = self._fit_nonexact(X_train, y_train)

    def predict(self, X_test):
        '''
        预测结果.

        参数:
            X_test: 预测用的数据集

        返回:
            y_predict: 预测的标签值
        '''
        if self.w is None or self.b is None:
            return None
        y_predict = self._predict(X_test, self.w, self.b).reshape(-1,1)
        return y_predict
