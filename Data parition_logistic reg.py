"""
Created on Wed Mar 13 17:57:57 2024
"""

# step1: import the files
import numpy as np
import pandas as pd
df  = pd.read_csv("breast_cancer.csv")

# step2: EDA
df.shape
df.head()

# step3: Data Cleaning
# step4: Data Transformation

# label encoding
df['Class']

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Class'] = LE.fit_transform(df['Class'])
df.head()

# standadization
df.iloc[:, 1:10].head()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df.iloc[:, 1:10] = SS.fit_transform(df.iloc[:, 1:10])
df.iloc[:, 1:10].head()

Y = df["Class"]
X = df.iloc[:, 1:10]

#==============================================================

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test   = train_test_split(X,Y,test_size=0.3)

#==============================================================

# step5: Model fitting
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

# predictions
Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

from sklearn.metrics import accuracy_score

ac1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy score:", ac1.round(2))

ac2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy score:", ac2.round(2))

#==============================================================
# cross validation --> validation set appraoch
#==============================================================

train_acc = []
test_acc = []

for i in range(1,101,1):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=i)
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_pred_train))
    test_acc.append(accuracy_score(Y_test,Y_pred_test))

print("Cross validation training accuracy:",np.mean(train_acc).round(2))
print("Cross validation test accuracy:",np.mean(test_acc).round(2))

#------------------------------------------------------------------------------

#==============================================================
# cross validation --> K-Fold cross validation
#==============================================================

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5) # by default shuffle = False

train_acc = []
test_acc = []

for train_index, test_index in kf.split(X):
    X_train, X_test, Y_train, Y_test = X.iloc[train_index],X.iloc[test_index],Y.iloc[train_index],Y.iloc[test_index]
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    train_acc.append(accuracy_score(Y_train,Y_pred_train))
    test_acc.append(accuracy_score(Y_test,Y_pred_test))

print("Cross validation training accuracy:",np.mean(train_acc).round(2))
print("Cross validation test accuracy:",np.mean(test_acc).round(2))


'''
X.iloc[:,:]
X.iloc[train_index]
type(train_index)

t1 = np.array([1,2,3,4,7,8,9])
t1
X.iloc[t1]

'''















