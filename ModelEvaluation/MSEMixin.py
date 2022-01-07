class MSEMixin:
    '''
    计算均方误差的基类
    '''

    def MSE_sum(self, X, Y):
        '''
            计算X,y误差的平方和

        参数：
            X:第一个数据集
            Y:第二个数据集
        '''
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        distance = X-Y
        return distance.T.dot(distance)[0, 0]

    def MSE(self, X, Y):
        '''
        计算X,y误差的平方和

        参数：
            X:第一个数据集
            Y:第二个数据集
        '''
        MSE_sum = self.MSE_sum(X, Y)
        m, n = X.shape
        return MSE_sum/(m*n)
