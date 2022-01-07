import numpy as np
from scipy.stats import f
import math
from ModelEvaluation.MSEMixin import MSEMixin


class RegressionUtility(MSEMixin):
    '''
    辅助工具基类

    参数:
        f_test_confidence_interval: F检验的置信度
    '''
    f_test_confidence_interval = None

    def _set_f_test_confidence_interval(self, f_test_confidence_interval):
        self.f_test_confidence_interval = 0.95
        if isinstance(f_test_confidence_interval, float) and f_test_confidence_interval > 0 and f_test_confidence_interval < 1:
            self.f_test_confidence_interval = f_test_confidence_interval

    def _forward_f_test(self, MSE_A, MSE_min, m, confidence_interval):
        '''
        逐步向前F检验

        参数:
            confidence_interval 置信度
            m: F分布的自由度
            MSE_min: 加入特征组后的最小MSE
            MSE_A: 此前的MSE
        '''
        if MSE_min > MSE_A:
            return False
        F = MSE_A/MSE_min
        p_value = f.cdf(F, m, m)
        return p_value > confidence_interval

    def _backward_f_test(self, MSE_A, MSE_min, m, confidence_interval):
        '''
        向后逐步的F检验

        参数:
            confidence_interval 置信度
            m: F分布的自由度
            MSE_min: 去掉特征组后的最小MSE
            MSE_A: 此前的MSE
        '''
        if MSE_min < MSE_A:
            return True
        F = MSE_min/MSE_A
        p_value = f.cdf(F, m, m)
        return p_value <= confidence_interval

    def _get_subset(self, Xy, sub_rate, yn):
        '''
        获取采样

        参数:
            Xy: 数据集
            subrate: 采样率
            yn: y的标签个数
        '''
        m, n = Xy.shape
        np.random.shuffle(Xy)
        return Xy[:math.ceil(m*sub_rate), 0:n-yn], Xy[:math.ceil(m*sub_rate), n-yn:n]

    def _check_passed(self, X_subset, y_subset, y_subset_predict, max_distance, min_pass_num):
        '''
        检查是否通过采样检验,并且选取合格的数据

        参数:
            X_subset:
            y_subset: 采样结果集.
            y_subset_predict: 采样预测集
            max_distance: 允许的误差范围
            min_pass_num: 更新结果需要的的最小个数
            min_MSE: 此前计算出的最小均方误差

        返回:
            是否通过检验
            X_new: 新的特征子集
            y_new: 新的结果子集

        '''
        m, n = y_subset.shape
        sum_count = m
        pass_count = 0
        y_new = []
        X_new = []
        for i in range(m):
            if(abs(y_subset[i]-y_subset_predict[i]) < max_distance):
                pass_count += 1
                y_new.append(y_subset[i])
                X_new.append(X_subset[i])
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        return pass_count >= min_pass_num, X_new, y_new
