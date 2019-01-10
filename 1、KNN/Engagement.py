'''
匹配约会对象
'''
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
from Knn import *
'''
文本转为numpy矩阵
'''


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))  # 构造行数=文本行数，列数=特征数的numpy零矩阵
    classLabelVector = []  # 标签
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 选取前三个字符将他们存储到特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  # 将文本中的最后一列(列表标签)单独存储到一个列表中
        index += 1
    return returnMat, classLabelVector
'''
归一化特征
dataSet：训练集样本(特征矩阵)
newValue = (oldValue - min)/(maxValue - minValue)
'''
def autoNorm(dataset):
    minVals = np.min(dataset,axis=0)#每一列最小值
    maxVals = np.max(dataset,axis=0)#每一列的最大值
    ranges = maxVals-minVals #每一列的取值范围
    normDataSet  = np.zeros(np.shape(dataset))#构造一个新的dataSet大小的numpy零矩阵
    m = np.size(dataset, axis=0)
    normDataSet  = dataset - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

'''
测试
'''
def datingClassTest():
    hoRadio=0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = np.size(normMat,axis=0)
    numTestVecs = int(hoRadio*m)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("The classifier came back with:%d,the real answer is:%d" % (classifierResult,datingLabels[i]))
        if classifierResult !=  datingLabels[i]:
            errorCount+=1
    print("the total error rate is:%f" % (errorCount / float(numTestVecs)))

def classifyPerson():
    resultList = ["not at all", "in small does", "in large does"]
    precentTaps = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumes per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,precentTaps,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,datingDataMat,datingLabels,3)
    print("You will probably like this person:",resultList[classifierResult - 1])

# datingClassTest()
# classifyPerson()