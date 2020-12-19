
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

# Lấy dữ liệu ảnh
trainData = pd.read_csv('digit-recognizer/train.csv')

# Hiển thị phân bố dữ liệu
sns.countplot(trainData['label'])
plt.show()

# Tách dữ liệu ảnh và nhãn
x_train_data = (trainData.iloc[:,1:].values).astype('float32')
y_train_label = (trainData.iloc[:,0].values).astype('float32')

# Hiển thị một ảnh bất kì
plt.imshow(x_train_data[1].reshape(28,28), cmap=plt.cm.binary)
plt.show()

# Đưa dữ liệu về [0, 1]
x_train_data = x_train_data/255.0


from sklearn.model_selection import train_test_split

# Chia dữ liệu ra tập train và test
X_train, X_test, Y_train, Y_test = train_test_split(x_train_data,y_train_label,test_size=0.3)

# Train KNN
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors=5,p=2,weights='distance')
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print('Accuracy with KNN: ',100*accuracy_score(Y_test,y_pred))

# Train SVC
from sklearn.svm import LinearSVC
clf_svc = LinearSVC()
clf_svc.fit(X_train, Y_train)
predicted = clf_svc.predict(X_test)
print('Accuracy_score SVC: ',100*accuracy_score(Y_test,predicted))


# from sklearn.metrics import confusion_matrix,classification_report

# Hiển thị confusion matrix KNN
# cm = confusion_matrix(Y_test,y_pred)
# f, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot= True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()

# Hiển thị confusion matrix SVC
# cm = confusion_matrix(Y_test,predicted)
# f, ax = plt.subplots(figsize=(5,5))
# sns.heatmap(cm,annot= True, linewidths= 0.5, linecolor="red", fmt=".0f", ax=ax)
# plt.xlabel("y_pred")
# plt.ylabel("y_true")
# plt.show()

# Tim gia tri k hop li
# X_train, X_test, Y_train, Y_test = train_test_split(x_train_data,y_train_label,test_size=5000, train_size=10000)
error = []
for i in  range(2,10):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i,p=2)
    knn.fit(X_train,Y_train)
    y_pred_i = knn.predict(X_test)
    error.append(np.mean(y_pred_i != Y_test))
plt.figure(figsize=(12,6))
plt.plot(range(2,10),error,color="red",linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error rate for K value')
plt.xlabel('K value')
plt.ylabel('Mean Error')
plt.show()

# def myweight(distances):
#     sigma2 = .5 # we can change this number
#     return np.exp(-distances**2/sigma2)
#
# clf_f = neighbors.KNeighborsClassifier(n_neighbors = 11, p = 2, weights = myweight)
# clf_f.fit(X_train, Y_train)
# y_pred = clf_f.predict(X_test)
#
# print("Accuracy of fomula: " ,(100*accuracy_score(Y_test, y_pred)))

