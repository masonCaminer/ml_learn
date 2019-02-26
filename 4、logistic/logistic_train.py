import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt', encoding='utf-8') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMat, labelMat):
    """
    梯度上升算法
    :param dataMat:list
    :param labelMat: list
    :return:[w0,w1,w2]
    """
    data_matrix = np.mat(dataMat, dtype=np.float64)
    label_matrix = np.mat(labelMat).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(data_matrix * weights)
        # 梯度上升矢量化公式
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
        # 将矩阵转为i数组，并返回权重
    return weights.getA()


"""
函数说明：绘制数据集

"""


def plotBestFit(weights):
    # 加载数据集
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    # 数据个数
    n = np.shape(dataMat)[0]
    # 正样本
    xcord1 = []
    ycord1 = []
    # 负样本
    xcord2 = []
    ycord2 = []
    # 根据数据集标签进行分类
    for i in range(n):
        # 1为正样本
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        # 0为负样本
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    # 绘制负样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    x = np.arange(-3.0, 3.0, 0.1)

    y = (-weights[0] - weights[1] * x) / weights[2]

    ax.plot(x, y)

    plt.title('BestFit')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = gradAscent(dataMat, labelMat)
    print(weights)
    plotBestFit(weights)
