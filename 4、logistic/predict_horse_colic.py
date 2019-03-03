"""
预测病马的死亡率

使用可用特征的均值来填补缺失值；
使用特殊值来填补缺失值，如-1；
忽略有缺失值的样本；
使用相似样本的均值添补缺失值；
使用另外的机器学习算法预测缺失值。

@Auther : Mason
@Date   : 
"""
import numpy as np
import random

import numpy as np
import random

"""
函数说明：sigmoid函数
"""


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


"""
函数说明：改进的梯度上升算法

"""


def stocGradAscent1(dataMatix, classLabels, numIter=500):
    m, n = np.shape(dataMatix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 选择随机选取的样本，计算h
            h = sigmoid(sum(dataMatix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatix[randIndex]
            # 删除已经使用的样本
            del (dataIndex[randIndex])
    return weights


"""
函数说明：使用python写的Logistic分类器做预测

"""


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    # 使用改进的随即上升梯度训练
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)


"""
函数说明：分类函数

"""


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    colicTest()
