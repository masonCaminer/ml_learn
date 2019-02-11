import numpy as np
import matplotlib.pyplot as plot


def loadDataSet():
    """

    :return:
    """
    dataMat = []
    labelMat = []
    with open('testSet.txt', 'r', encoding='utf-8') as lines:
        for line in lines:
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def plotDataSet(weights):
    # 加载数据集
    dataeMat, labelMat = loadDataSet()
    dataArr = np.array(dataeMat)
    # 数据个数 　
    n = np.shape(dataeMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本,坐标
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plot.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)

    plot.title('DataSet')
    plot.xlabel('x1')
    plot.ylabel('x2')
    plot.show()


def stocGradAscent1(dataMat, labelMat, numlter=150):
    """
    随机梯度上升算法
    :param dataMat:
    :param labelMat:
    :param numlter:
    :return:
    """
    m, n = np.shape(dataMat)
    # 参数初始化
    weights = np.ones(n)
    for j



def gradAscent(dataMat, labelMat):
    """
    梯度上升算法
    :param dataMat: 数据
    :param labelMat: 标签
    :return:
    """
    # 矩阵
    dataMatrix = np.mat(dataMat)
    # 矩阵，转置
    labelMat = np.mat(labelMat).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmod(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转化为数组，并返回权重系数
    return weights.getA()


def sigmod(inX):
    """
    sigmod公式
    :param inX:
    :return:
    """
    return 1.0 / (1 + np.exp(-inX))


if __name__ == '__main__':
    # plotDataSet()
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    plotDataSet(weights)
