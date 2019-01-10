from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
a = le.fit([1,5,67,100])
a = le.transform([1,5,67,100])
# a = le.fit_transform([1,1,100,67,5])
print(a)