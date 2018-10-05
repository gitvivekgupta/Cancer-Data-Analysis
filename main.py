import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('Meddata.csv', header=0)
data = data.dropna()
# print(data.shape)
# print(list(data.columns))
# sns.countplot(x='Class',data=data, palette='hls')
# plt.show()

# print(data.isnull().sum())

sns.countplot(y="Mitoses", data=data)
plt.show()

exit(0)

data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)



X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)


print(classification_report(y_test, y_pred))


