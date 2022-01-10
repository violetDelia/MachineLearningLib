from enum import Enum
import numpy as np
from ModelEvaluation.ModelEvaluation import ModelEvaluation
from DataVisualization.DateVisualization import DataVisualization
from Preprocessing.Preprocessing import Preprocessing
from LinearRegression.RidgeRegression import RidgeRegression
from LinearRegression.RegressionUtility import RegressionUtility
from LinearRegression.StagewiseRegression import StagewiseRegression
from LinearRegression.LassoRegression import LassoRegression


class LinearRegression(RidgeRegression, StagewiseRegression, LassoRegression, DataVisualization, Preprocessing, ModelEvaluation, RegressionUtility):
    '''
    线性回归模型

    属性:
        Lambda_l2: 二范数正则化系数
        w: 各个特征的权重
        A: 选取的特征组
            用于逐步回归
        f_test_confidence_interval: F检验的置信度
        stagewize_learning_rate: 分段回归算法中的学习率
        stagewize_max_steps: 分段回归迭代次数
        GD_max_steps: 梯度下降最大迭代次数
        GD_step_rate: 梯度下降搜索步长
        GD_epsilon: 梯度下降结果误差许可值
        GD_init_w: 梯度下降的初始值
    '''
    A = None

    class SoulutionType(Enum):
        '''
        求解类型:

        normal: 正规方程求解
        GD: 梯度下降法求解
        SGD: 次梯度下降求解
        '''
        normal = 0
        '''
        正规方程求解
        '''
        GD = 1
        '''
        梯度下降求解
        '''
        SGD = 2
        '''
        次梯度下降求解
        '''

    class ProcessingType(Enum):
        '''
        特征处理的类型

        not_process: 不处理
        normal: 常规处理
        multinomial: 多项式处理
        '''
        not_process = 0
        '''
        不处理
        '''
        normal = 1
        '''
        常规处理
        '''
        multinomial = 2
        '''
        多项式处理
        '''

    class RegressionType(Enum):
        '''
        回归类型

        LinearRegression: 普通线性回归
        RigidRegression: 岭回归
        StagewiseRegression: 分段回归 (根据相关系数)
        LassoRegression: lasso回归
        '''
        LinearRegression = 0
        '''
        普通线性回归
        '''
        RidgeRegression = 1
        '''
        岭回归
        '''
        StagewiseRegression = 2
        '''
        分段回归 (根据相关系数)
        '''
        LassoRegression = 3
        '''
        lasso回归
        '''

    class FeatureSelectType(Enum):
        '''
        特征选择类型
            step_forward: 向前逐步
            step_backward: 向后逐步
            stage_wise: 分段回归
        '''
        step_forward = 0
        '''
        向前逐步
        '''
        step_backward = 1
        '''
        向后逐步
        '''

    def __init__(self, Lambda_l1=1, Lambda_l2=1,
                 f_test_confidence_interval=0.95, stagewize_learning_rate=0.01, stagewize_max_steps=10000,
                 GD_max_steps=1000, GD_step_rate=0.1, GD_epsilon=0.001, GD_init_w=None):
        self._set_Lambda_l1(Lambda_l1)
        self._set_Lambda_l2(Lambda_l2)
        self._set_f_test_confidence_interval(f_test_confidence_interval)
        self._set_stagewize_learning_rate(stagewize_learning_rate)
        self._set_stagewize_max_steps(stagewize_max_steps)
        self._set_GD_max_steps(GD_max_steps)
        self._set_GD_step_rate(GD_step_rate)
        self._set_GD_epsilon(GD_epsilon)
        self._set_GD_init_w(GD_init_w)

    def _preprocessing(self, X_train, y_train, processingType, weights, processing_feature_degree):
        '''
        数据集预处理

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
            processing_feature_degree: 多项式处理的次数
        '''

        if processingType == self.ProcessingType.not_process:
            X_processed = X_train
        elif processingType == self.ProcessingType.normal:
            X_processed = self._add_onevector_to_features(X_train)
        elif processingType == self.ProcessingType.multinomial:
            X_processed = self._process_feature_to_multinomial_features(
                X_train, processing_feature_degree)
        if weights is not None:
            X_processed = self._process_weight_to_features(
                X_processed, weights)
        y_processed = y_train
        return X_processed, y_processed

    def _prepare(self, X_train, y_train, regressionType, soulutionType, processingType, weights,
                 processing_feature_degree, featureSelectionType):
        '''
        数据处理以及特征选取

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            regressionType: 回归的类型
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
            processing_feature_degree: 多项式处理时的次数
            featureSelectionType: 特征选择类型
        '''
        if self._check_type(regressionType, soulutionType, processingType, featureSelectionType) is False:
            print("类型参数错误!")
            return
        X_processed, y_processed = self._preprocessing(
            X_train, y_train, processingType, weights, processing_feature_degree)
        m, n = X_processed.shape
        if featureSelectionType is not None:
            if featureSelectionType == self.FeatureSelectType.step_forward:
                self._forward_selection(
                    X_processed, y_processed, regressionType, soulutionType)
            elif featureSelectionType == self.FeatureSelectType.step_backward:
                self._backward_selection(
                    X_processed, y_processed, regressionType, soulutionType)
        return X_processed, y_processed

    def _check_type(self, regressionType, soulutionType, processingType, selectionType):
        '''
        检查类型参数是否错误

        参数:
            soulutionType: 求解类型
            processingType: 数据处理类型
            regressionType: 回归的类型
            selectionType: 特征选择模型
        '''
        if isinstance(processingType, self.ProcessingType) and isinstance(soulutionType, self.SoulutionType) and isinstance(regressionType, self.RegressionType):
            if selectionType is None:
                return True
            else:
                if isinstance(selectionType, self.FeatureSelectType):
                    return True
        return False

    def _forward_selection(self, X_processed, y_processed, regressionType, soulutionType):
        '''
        逐步向前选择特征:

        参数:
            X_processed: 处理好的特征集
            y_processed: 处理好的结果集
            regressionType: 回归的类型
            soulutionType: 求解类型
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        m, n = X_processed.shape
        A, C = [0], [i for i in range(1, n)]
        self.A = A
        if regressionType == self.RegressionType.StagewiseRegression:
            regressionType = self.RegressionType.LinearRegression
        while len(C) > 0:
            w = self.fit(X_processed, y_processed, regressionType, soulutionType,
                         self.ProcessingType.not_process, None, None, None)
            distances = y_processed - X_processed[:, A].dot(w)
            MSE_A = distances.T.dot(distances)[0, 0]
            MSE_min = float("inf")
            j_min = -1
            for j in C:
                self.A = A+[j]
                w = self.fit(X_processed, y_processed, regressionType, soulutionType,
                             self.ProcessingType.not_process, None, None, None)
                distances = y_processed - X_processed[:, A+[j]].dot(w)
                MSE_j = distances.T.dot(distances)[0, 0]
                if MSE_j < MSE_min:
                    MSE_min, j_min = MSE_j, j
            if self._forward_f_test(MSE_A, MSE_min, m, self.f_test_confidence_interval):
                A.append(j_min)
                C.remove(j_min)
            else:
                break
        print(A)
        self.A = A

    def _backward_selection(self, X_processed, y_processed, regressionType, soulutionType):
        '''
        逐步向后选择特征:

        参数:
            X_processed: 处理好的特征集
            y_processed: 处理好的结果集
            regressionType: 回归的类型
            soulutionType: 求解类型
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        m, n = X_processed.shape
        A, C = [i for i in range(0, n)], []
        self.A = A
        if regressionType == self.RegressionType.StagewiseRegression:
            regressionType = self.RegressionType.LinearRegression
        while len(A) > 1:
            w = self.fit(X_processed, y_processed, regressionType, soulutionType,
                         self.ProcessingType.not_process, None, None, None)
            distances = y_processed - X_processed[:, A].dot(w)
            MSE_A = distances.T.dot(distances)[0, 0]
            MSE_min = float('inf')
            j_min = -1
            for j in A:
                A_copy = A.copy()
                A_copy.remove(j)
                self.A = A_copy
                w = self.fit(X_processed, y_processed, regressionType, soulutionType,
                             self.ProcessingType.not_process, None, None, None)
                distances = y_processed - X_processed[:, A_copy].dot(w)
                MSE_j = distances.T.dot(distances)[0, 0]
                if MSE_j < MSE_min:
                    MSE_min, j_min = MSE_j, j
            if self._backward_f_test(MSE_A, MSE_min, m, self.f_test_confidence_interval):
                A.remove(j_min)
                C.append(j_min)
            else:
                break
        self.A = A

    def RANSAC(self, X_train, y_train, max_distance, pass_rate=0.6, sub_rate=0.3, max_steps=100,
               regressionType=RegressionType.LinearRegression, soulutionType=SoulutionType.normal,
               processingType=ProcessingType.not_process, weights=None,
               processing_feature_degree=2, featureSelectionType=None):
        '''
        随机采样

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            regressionType: 回归的类型
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
            processing_feature_degree: 多项式处理时的次数
            featureSelectionType: 特征选择类型
            max_distance: 允许的误差范围
            pass_rate: 超过这个比率的点在误差范围内就挑选出合格的样本点再一次计算模型并更新
            sub_rate: 选取子集的比例
            max_steps: 迭代数
        注意:
            如果没有w值增加max_distance
        '''
        X_processed, y_processed = self._prepare(X_train, y_train, regressionType, soulutionType, processingType,
                                                 weights, processing_feature_degree, featureSelectionType)
        step = 0
        min_MSE = float("inf")
        best_w = 0
        ym, yn = y_processed.shape
        Xy = np.concatenate((X_processed, y_processed), axis=1)
        while step < max_steps:
            X_subset, y_subset = self._get_subset(Xy, sub_rate, yn)
            current_w = self.fit(X_subset, y_subset, regressionType, soulutionType, self.ProcessingType.not_process,
                                 weights, processing_feature_degree, None)
            y_subset_predict = self._predict(
                X_subset, current_w, self.ProcessingType.not_process, processing_feature_degree, weights)
            need_recompute, X_new, y_new = self._check_passed(
                X_subset, y_subset, y_subset_predict, max_distance, pass_rate)
            if need_recompute:
                current_w = self.fit(X_new, y_new, regressionType, soulutionType, self.ProcessingType.not_process,
                                     weights, processing_feature_degree, None)
                y_new_predict = self._predict(
                    X_new, current_w, self.ProcessingType.not_process, processing_feature_degree, weights)
                current_MSE = self.MSE(y_new, y_new_predict)
                # y_predict = self._predict(
                #     X_processed, current_w, self.ProcessingType.not_process, processing_feature_degree,weights)
                # current_MSE = self.MSE(y_processed, y_predict)
                if current_MSE < min_MSE:
                    min_MSE = current_MSE
                    best_w = current_w
            # else:
            #     current_MSE = self.MSE(y_subset, y_subset_predict)
            #     if current_MSE < min_MSE:
            #         min_MSE = current_MSE
            #         best_w = current_w
            step += 1
        self.w = best_w

    def fit(self, X_train, y_train, regressionType, soulutionType, processingType, weights, processing_feature_degree, featureSelectionType):
        '''
        寻找w

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            regressionType: 回归的类型
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
            processing_feature_degree: 多项式处理时的次数
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        X_processed, y_processed = self._prepare(X_train, y_train, regressionType, soulutionType, processingType,
                                                 weights, processing_feature_degree, featureSelectionType)
        m, n = X_processed.shape
        if self.A is None:
            self.A = [i for i in range(0, n)]
        X_processed = X_processed[:, self.A]
        if regressionType == self.RegressionType.LinearRegression:
            if soulutionType == self.SoulutionType.normal:
                return self._fit_linear_normal(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.GD:
                return self._fit_linear_GD(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.SGD:
                return None
        elif regressionType == self.RegressionType.RidgeRegression:
            if soulutionType == self.SoulutionType.normal:
                return self._fit_ridge_normal(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.GD:
                return self._fit_ridge_GD(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.SGD:
                return None
        elif regressionType == self.RegressionType.StagewiseRegression:
            if soulutionType == self.SoulutionType.normal:
                return self._fit_stagewize(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.GD:
                return self._fit_stagewize(X_processed, y_processed)
            elif soulutionType == self.SoulutionType.SGD:
                return self._fit_stagewize(X_processed, y_processed)
        elif regressionType == self.RegressionType.LassoRegression:
            if soulutionType == self.SoulutionType.normal:
                return None
            elif soulutionType == self.SoulutionType.GD:
                return None
            elif soulutionType == self.SoulutionType.SGD:
                return self._fit_lasso_SGD(X_processed, y_processed)

    def train(self, X_train, y_train, regressionType=RegressionType.LinearRegression,
              soulutionType=SoulutionType.normal, processingType=ProcessingType.normal,
              weights=None, processing_feature_degree=2, featureSelectionType=None,):
        '''
        训练模型

        参数:
            X_train: 训练用的样本,是一个(m,n)的数组.
            y_train: 标签向量,是一个(m,1)的向量
            regressionType: 回归的类型
            soulutionType: 求解类型
            processingType: 数据处理类型
            weights: 权重向量
            processing_feature_degree: 多项式处理时的次数
            featureSelectionType: 特征选择类型
            GD_max_steps: 梯度下降最大迭代次数
            GD_step_rate: 梯度下降搜索步长
            GD_epsilon: 梯度下降结果误差许可值
        '''
        self.w = self.fit(X_train, y_train, regressionType, soulutionType,
                          processingType, weights, processing_feature_degree, featureSelectionType)

    def predict(self, X, processingType=ProcessingType.normal, processing_feature_degree=2, weights=None):
        '''
        用类属性w预测X对应的y

        参数:
            X: 预测用的特征集
            weights: 权重向量
            processingType: 数据处理类型
            processing_feature_degree: 多项式处理时的次数
        '''
        X_processed, y_processed = self._preprocessing(
            X, None, processingType, weights, processing_feature_degree)
        return X_processed[:, self.A].dot(self.w)

    def _predict(self, X, w, processingType=ProcessingType.normal, processing_feature_degree=2, weights=None):
        '''
        用w预测X对应的y

        参数:
            X: 预测用的特征集
            w: w向量
            weights: 权重向量
            processingType: 数据处理类型
            processing_feature_degree: 多项式处理时的次数
        '''
        X_processed, y_processed = self._preprocessing(
            X, None, processingType, weights, processing_feature_degree)
        return X_processed[:, self.A].dot(w)
