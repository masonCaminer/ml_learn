import numpy as np
import matplotlib.pyplot as plt
import random


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
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(data_matrix * weights)
        # 梯度上升矢量化公式
        error = label_matrix - h
        weights = weights + alpha * data_matrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    # 将矩阵转为i数组，并返回权重
    return weights.getA(), weights_array


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    随机梯度上升
    （一）
    alpha在每次迭代的时候都会进行调整，虽然每次都会减小，但是不会减小到0，因为此处有常数项。

    如果需要处理的问题是动态变化的，那么可以适当加大上述常数项，来确保新的值获得更大的回归系数。

    alpha减少的过程中，每次减小1/(j+i)1/(j+i)，其中jj为迭代次数，ii为样本点的下标。

    （二）

    更新回归系数时，只使用一个样本点，并且选择的样本点是随机的，每次迭代不使用已经用过的样本点，减小了计算量，并且保证了回归效果。

    :param dataMatrix:
    :param classLabels:
    :param numIter:
    :return:
    """
    dataMatrix = np.array(dataMatrix)
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    weights_array = np.array([])
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        # 参数初始化
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机获取样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 选择随机选取的一个样本，计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 更新回归系数
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 添加回归系数到数组中
            weights_array = np.append(weights_array, weights, axis=0)
            del (dataIndex[randIndex])
    # 改变维度
    weights_array = weights_array.reshape(numIter * m, n)
    return weights, weights_array


def plotBestFit(weights):
    """
    函数说明：绘制数据集

    :param weights:
    :return:
    """
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


def plotWeights(weights_array1, weights_array2):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title('梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][0].set_ylabel('W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel('W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel('迭代次数')
    axs2_ylabel_text = axs[2][0].set_ylabel('W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title('改进的随机梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][1].set_ylabel('W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel('W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel('迭代次数')
    axs2_ylabel_text = axs[2][1].set_ylabel('W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(dataMat, labelMat)
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotBestFit(weights1)
    plotBestFit(weights2)
    plotWeights(weights_array1, weights_array2)
