"""

@Auther : Mason
@Date   : 
"""
from sklearn.linear_model import LogisticRegression

"""
函数说明：使用sklearn构建logistics分类器

"""


def colicSklearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open(('horseColicTest.txt'))
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []

    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))

    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))

    classifier = LogisticRegression(solver='liblinear', max_iter=10000).fit(trainingSet, trainingLabels)
    test_accuracy = classifier.score(testSet, testLabels) * 100
    print('正确率：%f%%' % test_accuracy)


if __name__ == '__main__':
    colicSklearn()
