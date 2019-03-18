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


def plotDataSet():
    # 加载数据集
    dataeMat, labelMat = loadDataSet()
    dataArr = np.array(dataeMat)
    # 数据个数 　
    n = np.shape(dataeMat)[0]
    #正样本
    xcord1= [];ycord1 = []
    #负样本
    xcord2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[1])
            ycord1.append(dataArr[2])
        else:
            xcord2.append(dataArr[1])
            ycord2.append(dataArr[2])
    fig = plot.figure()
    # 添加subplot
    ax = fig.add_subplot(111)
    # 绘制正样本
    ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=5)



if __name__ == '__main__':
    plotDataSet()
