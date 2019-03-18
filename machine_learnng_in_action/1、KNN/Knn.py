from numpy import *
import operator
import numpy as np


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'C', 'D']
    return group, labels


'''
kNN算法核心分类函数
这里距离使用的是欧式距离sqrt((a1-b1)**2 + (a2-b2)**2 + ... + (an-bn)**2)
参数：
inX：用于分类的输入向量
dataSet：训练集样本
labels：标签向量
k：选择最近邻居的数目
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = np.size(dataSet, axis=0)  # 数据集样本数目
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile函数构建dataSet同样规格的矩阵，输入向量的各个特征减去训练集中对应特征
    sqDiffMat = diffMat ** 2  # 计算输入向量与训练集中各个样本的距离
    sqDistances = np.sum(sqDiffMat, axis=1)  # 把每一列的矩阵相加
    sortedIndicies = np.argsort(sqDistances)  # 按照距离升序，新矩阵显示的是在之前矩阵的距离
    classCount = {}  # 存储k个最近邻居的标签及数目
    for i in range(k):  # 找到k个最近邻居并存储
        voteILabel = labels[sortedIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 对value降序排序，返回一个数组
    return sortedClassCount[0][0]  # 预测的分类


group, labels = createDataSet()
classify0([0, 0], group, labels, 3)
