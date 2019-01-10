import numpy as np
from functools import reduce
def loadDataSet():
    # 词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 标签向量，1侮辱，0不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    """
    将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    :param dataSet:整理的样本数据集
    :return:返回不重复的词条列表，也就是词汇表
    """
    #创建一个空的不重复列表
    vocabSet = set([])
    for document in dataSet:
        # 取并集
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(myVocabList,postingList):
    """
    将inputSet向量化，向量的每个元素为1或0
    :param myVocabList:createVocabList返回的列表
    :param postingList:切分的词条列表
    :return:文档向量，词集模型
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0]*len(myVocabList)
    for word in postingList:
        if word in myVocabList:
            returnVec[myVocabList.index(word)]=1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMtrix,trainCategory):
    # 计算训练的文档数
    numTrainDocs = len(trainMtrix) # 6
    # 计算每篇文章的词条数
    numWords = len(trainMtrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs) #0.5
    # 创建numpy.zeros数组,词条初始化为1，拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为2.0,拉普拉斯平滑
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率,即P(w0|1),P(w1|1),P(w2|1)···
        if trainCategory[i]==1:
            p1Num += trainMtrix[i]
            p1Denom += sum(trainMtrix[i])
        else:
            #统计属于非侮辱类的条件概率，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num +=trainMtrix[i]
            p0Denom += sum(trainMtrix[i])
    # 相除
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    朴素贝叶斯分类器分类函数
    :param vec2Classify:待分类的词条数组
    :param p0Vec:侮辱类的条件概率数组
    :param p1Vec:非侮辱类的条件概率数组
    :param pClass1:文档属于侮辱类的概率
    :return:0 ：属于非侮辱类
            1 ：属于侮辱类
    """
    # 对应元素相乘,logA*B=logA+logB，所以要加上np.log(pClass1)
    # p1 = reduce(lambda x, y: x * y, vec2Classify * p1Vec) * pClass1
    # p0 = reduce(lambda x, y: x * y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print(p0)
    print(p1)
    if p1>p0:
        return 1
    else:
        return 0
if __name__ == "__main__":
    postingList, classVec = loadDataSet()
    print('postingList:\n',postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print('trainMat:\n', trainMat)
    """
    p0V存放的是每个单词属于类别0，也就是非侮辱类词汇的概率 
    p1V存放的就是各个单词属于侮辱类的条件概率。pAb就是先验概率。
    """
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
    # 测试样本1
    testEntry = ['love', 'my', 'dalmation']
    # 测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList,testEntry))
    # 分类
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry, '属于侮辱类')
        # 执行分类并打印分类结果
    else:
        print(testEntry, '属于非侮辱类')