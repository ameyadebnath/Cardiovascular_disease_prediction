#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import pickle

#importing the csv file
data = pd.read_csv('cardio_train.csv', delimiter=';')

data.head()

#Data modification
#1. Age - Converting age into years
data['age']=data['age']/365

#Checking for any missing value
data.isnull().sum()

X = data.drop(['cardio', 'id'], axis=1)
y = data['cardio']

#Train test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Shape of x_train: ', x_train.shape)
print('Shape of x_test: ', x_test.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of y_test: ', y_test.shape)

#Standardising data
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

#save standardise as pickle
pickle.dump(scaler, open("scaler.pkl", 'wb'))

gamma = [0.01, 0.1, 1, 10, 100]
weight = [None, 'balanced']
param_distributions = dict(C=gamma, class_weight=weight)
print(param_distributions)

# instantiate and fit the grid
grid = RandomizedSearchCV(LogisticRegression(penalty='l2'), param_distributions, cv=5, scoring='f1', return_train_score=False, n_jobs=-1)

grid.fit(x_train_s, y_train)

# examine the best model
print(grid.best_score_)
print(grid.best_params_)

LR_optimal=LogisticRegression(penalty='l2', C=100, class_weight='balanced')

# fitting the model
LR_optimal.fit(x_train_s, y_train)

#train
acc_log_train = round(LR_optimal.score(x_train_s, y_train) * 100, 2)

#test
acc_log_test = round(LR_optimal.score(x_test_s, y_test) * 100, 2)


acc_log_train

acc_log_test


#save model as pickle
pickle.dump(LR_optimal, open("classifier.pkl", "wb"))

#confusion matrix

dataset=pd.read_csv('cardio_train.csv', delimiter=';')
#dataset
x=pd.DataFrame(dataset.iloc[:,:-1])
y=pd.DataFrame(dataset.iloc[:,-1])
y_pred=LR_optimal.predict(x_test)

# actual = np.random.binomial(1,.9,size = 1000)
# predicted = np.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()
