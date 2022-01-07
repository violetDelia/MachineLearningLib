import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataVisualization:
    '''
    数据可视化的类
    '''

    def plot_true_scatter_and_predict_scatter(self, X_feature, y_true, y_predict, column=0, scatter_color="blue"):
        '''
        显示真实值和预测值的散点图

        参数:
            X_feature: 特征数据集
            y_true: 真实的结果集
            y_predict: 预测的结果集
            colum: 要显示的特征列
            scatter_color: 散点的颜色
        '''
        target_feature = X_feature[:, column]
        plt.scatter(target_feature, y_true)
        plt.scatter(target_feature, y_predict, color=scatter_color)

    def plot_true_scatter_and_predict_line(self, X_feature, y_true, y_predict, X_column=0, y_column=0, line_color="blue"):
        '''
        显示真实值的散点图和预测值的曲线

        参数:
            X_feature: 特征数据集
            y_true: 真实的结果集
            y_predict: 预测的结果集
            X_column: 要显示的特征列
            y_column: 要显示的结果列
            line_color: 直线的颜色
        '''
        target_feature = X_feature[:, X_column]
        target_y_predict = y_predict.reshape(-1, 1)[:, y_column]
        target_y_true = y_true.reshape(-1, 1)[:, y_column]
        feature_series = pd.Series(target_feature)
        predict_series = pd.Series(target_y_predict)
        data_dc = {"feature": feature_series, "predict": predict_series}
        data = pd.DataFrame(data_dc)
        data = data.sort_values(by=["feature"])
        plt.scatter(target_feature, target_y_true)
        plt.plot(data["feature"].values,
                 data["predict"].values, color=line_color)

    def plot_true_scatter_and_compare_predict_line(self, X_feature, y_true, y_predict1, y_predict2 ,X_column=0, y_column=0):
        '''
        显示真实值的散点图和并比较预测的曲线

        参数:
            X_feature: 特征数据集
            y_true: 真实的结果集
            y_predict1: 第一个预测的结果集
            y_predict2: 第二个预测的结果集
            X_column: 要显示的特征列
            y_column: 要显示的结果列
            line_color: 直线的颜色
        '''
        target_feature = X_feature[:, X_column]
        target_y_predict1 = y_predict1.reshape(-1, 1)[:, y_column]
        target_y_predict2 = y_predict2.reshape(-1, 1)[:, y_column]
        target_y_true = y_true.reshape(-1, 1)[:, y_column]
        feature_series = pd.Series(target_feature)
        predict1_series = pd.Series(target_y_predict1)
        predict2_series = pd.Series(target_y_predict2)
        data_dc = {"feature": feature_series,
                   "predict1": predict1_series, "predict2": predict2_series}
        data = pd.DataFrame(data_dc)
        data = data.sort_values(by=["feature"])
        plt.scatter(target_feature, target_y_true)
        plt.plot(data["feature"].values,
                 data["predict1"].values, color="red")
        plt.plot(data["feature"].values,
                 data["predict2"].values, color="blue")
