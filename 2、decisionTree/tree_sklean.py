import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    with open('lenses.txt', 'r', encoding='utf-8') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组类别
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    # 特征标签
    lensesLabel = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    for each_label in lensesLabel:
        for each in lenses:
            lenses_list.append(each[lensesLabel.index(each_label)])
        lenses_dict[each_label]=lenses_list
        lenses_list=[]
    print(lenses_dict)
    # 生成pandas.DataFrame
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    # 创建LabelEncoder()对象，用于序列化
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        lenses_pd[col]=le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    X = lenses_pd
    y= lenses_target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
