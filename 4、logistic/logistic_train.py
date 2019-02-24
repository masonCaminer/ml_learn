import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    with open('testSet.txt',encoding='utf-8') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, lineArr[0], lineArr[1]])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMat, labelMat):
    """
    梯度上升算法
    :param dataMat:list
    :param labelMat: list
    :return:
    """
    data_matrix = np.mat(dataMat)
    label_matrix = np.mat(labelMat).transpose()
    m, n = np.shape(data_matrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(data_matrix*weights)
        error = label_matrix-h
        weights = weights+alpha*data_matrix.transpose()*error
    return weights.getA()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    gradAscent(dataMat, labelMat)
