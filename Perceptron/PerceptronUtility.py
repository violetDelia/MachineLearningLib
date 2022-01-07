import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PerceptronUtility:

    def plot_2D_scatter_and_line(self,data, x_column, y_column, color_column, w, b,color = "blue"):
        '''
        绘制2D图像以及分离超平面

        参数:
            date:数据帧
            x_column:x轴的数据
            y_column:y轴的数据
            color_column:标签数据（区分颜色）
            w:感知机模型的w
            b:感知机模型的b
        '''
        line_x = np.arange(data[x_column].agg(np.min)-1,
                           data[x_column].agg(np.max)+1)
        w = w.reshape(1,-1)
        line_y = (-b-w[0][0]*line_x)/w[0][1]
        sns.relplot(x=x_column, y=y_column, hue=color_column, data=data)
        plt.plot(line_x, line_y,color = color)
