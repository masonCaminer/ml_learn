import numpy as np
import os
import operator
from  Engagement import *
'''
将32*32的黑白图像转换成1*1024的向量
'''


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):  # 循环读取每一个像素点(字符)，转换为1*1024的向量
        linestr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(linestr[j])
    return returnVect
'''
手写数字识别
'''
def handwritingClassTest():
    hwLabels =[]#标签
    trainingFileList = os.listdir("trainingDigits")
    m = len(trainingFileList)#统计训练样本数目
    trainingMat = np.zeros((m,1024))#构造m*1024numpy零矩阵，为了将所有训练样本转换成二维矩阵进行计算
    # 将所有训练样本转换成m*1024的numpy矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)#标签
        trainingMat[i,:]=img2vector('trainingDigits/'+fileNameStr)

    testFileList = os.listdir('testDigits')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/'+fileNameStr)
        prediction = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with: %d,the real answer is:%d" % (prediction, classNumStr))
        if classNumStr != prediction:
            errorCount+=1
    print("\nthe total number of errors is:%d" % errorCount)  # 预测错误次数
    print("\nthe total error rate is:%f" % (errorCount / float(mTest)))  # 预测错误率

handwritingClassTest()