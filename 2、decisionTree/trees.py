from math import log
import matplotlib.pyplot as plt
import operator
from matplotlib.font_manager import FontProperties
import pickle
"""
创建测试数据集
"""


def creatDataSet():
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataSet, labels


"""
计算信息熵
"""


def calcShannonEnt(dataSet):
    """
    计算信息熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)  # 数据集行数
    # 保存每个标签（label）出现次数的字典
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 经验熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


"""
计算给定数据集的经验熵（香农熵）
"""


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的特征的索引值
    :param dataSet:
    :return:
    """
    # 特征数量
    numFeatures = len(dataSet[0]) - 1
    # 计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优属性的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featlist = [example[i] for example in dataSet]
        uniqueVals = set(featlist)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            sunDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(sunDataSet) / float(len(dataSet))
            # 计算经验条件熵
            newEntropy += prob * calcShannonEnt((sunDataSet))
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
            # 返回信息增益最大特征的索引值
    return bestFeature


"""
    划分数据集
    dataSet：待划分的数据集
    axis：划分数据集的特征
    value：需要返回的特征的值
"""


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def majoriCnt(classList):
    """
    统计classList中出现次数最多的元素（类标签）
    :param classList:类标签列表
    :return:
    """
    classCount = {}
    # 统计classList中每个元素出现次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    """
    创建决策树
    :param dataSet: 训练数据集
    :param labels:分类属性标签
    :param featLabels:最优特征标签
    :return:
    """
    # 获取类别标签（是否放贷yes,no）
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majoriCnt(classList)
    # 选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 最优特征的标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已使用的特征标签
    del (labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [example[bestFeat] for example in dataSet]
    # 去掉重复的属性值
    uniqueVls = set(featValues)
    # 遍历特征，创建决策树
    for value in uniqueVls:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree

def getTreeDepth(myTree):
    """
    获取决策树层数
    :param myTree:
    :return: 决策树层数
    """
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDic = myTree[firstStr] #获取下一个字典
    for key in secondDic.keys():
        if type(secondDic[key]).__name__=='dict':
            thisDepth = 1+ getTreeDepth(secondDic[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:maxDepth=thisDepth
    return maxDepth

def getNumLeafs(myTree):
    """
    获取决策树叶节点个数，决定了树的宽度
    :param myTree:
    :return:
    """
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边属性值
    :param cntrPt:用于计算标注位置
    :param parentPt:用于计算标注位置
    :param txtString:标注的内容
    :return:
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    函数说明:绘制结点

    Parameters:
        nodeTxt - 结点名
        centerPt - 文本位置
        parentPt - 标注的箭头位置
        nodeType - 结点格式
    Returns:
        无
    """
    arrow_args = dict(arrowstyle="<-")                                            #定义箭头格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)        #设置中文字体
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',    #绘制结点
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree:决策树（字典）
    :param parentPt:标注的内容
    :param nodeTxt:节点名
    :return:
    """
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')  # 设置节点格式
    leafNode = dict(boxstyle='round4', fc='0.8')  # 设置叶节点格式
    numLeafs= getNumLeafs(myTree)#获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)#获取决策树层数
    firstStr = next(iter(myTree))#下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)#中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    """
    创建绘制面板
    :param inTree:决策树（字典）
    :return:
    """
    fig = plt.figure(1, facecolor='white')  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉想x,y轴
    plotTree.totalW = float(getNumLeafs(inTree))#获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))#获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0#x偏移
    plotTree(inTree, (0.5,1.0), '')#绘制决策树
    plt.show()#显示绘制结果

def classify(inputTree,featLabels,testVec):
    #获取决策树节点
    firstStr=next(iter(inputTree))
    #下一个字典
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)

    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabels,testVec)
            else: classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,fileName):
    """
    保存决策树
    :param inputTree: 已经生成的决策树
    :param fileName:存储文件名
    :return:
    """
    with open(fileName,'wb') as fw:
        pickle.dump(inputTree,fw)

def grabTree(filename):
    """
    读取决策树
    :param filename:
    :return:
    """
    fr = open(filename,'rb',encoding='utf-8')
    return pickle.load(fr)
# main函数
if __name__ == '__main__':
    dataSet, labels = creatDataSet()
    print(chooseBestFeatureToSplit(dataSet))
    print(dataSet)
    print(calcShannonEnt(dataSet))
    featLabels = []
    try:
        myTree =grabTree('classifierStorage.txt')
    except:
        myTree = createTree(dataSet, labels, featLabels)
    #     print(myTree)
    #     storeTree(myTree, 'classifierStorage.txt')
    # # createPlot(myTree)
    # # 测试数据
    testVec=[3,1]
    result = classify(myTree,featLabels,testVec)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')