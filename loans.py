# -*- coding: utf-8 -*-


import warnings
warnings.filterwarnings(action='once')

#!pip install feature_engine
#!pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import  export_graphviz
from sklearn.feature_selection import SelectFromModel
import graphviz

from feature_engine.encoding import OneHotEncoder
from feature_engine import transformation as vt
from feature_engine.wrappers import SklearnTransformerWrapper

import scipy.stats as st
import statsmodels.api as sm



from pathlib import Path

data_folder = Path("C:/Users/fridc/Documents/loans_project")

loan_data = data_folder/'loan_approval_dataset.csv'
df = pd.read_csv(loan_data)

df.head()

df.describe().transpose()

sns.countplot(x =' loan_status',data=df)

df = df.drop('loan_id',axis=1)

#for var in df:
#    print(f"Count plot for {var}")
 #   ax = sns.countplot(x=df[var], hue=df[' loan_status'])
 #   plt.show()

from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline
continuous = [
    var for var in df.columns if df[var].dtype.name == 'int64'
]

print(df[" loan_status"].head())

le = LabelEncoder()


df[" education"] = le.fit_transform(df[" education"])
df[" self_employed"] = le.fit_transform(df[" self_employed"])
df[" loan_status"] = le.fit_transform(df[" loan_status"])

print(df[" loan_status"].head())





x_train, x_test, y_train, y_test = train_test_split(df.drop(' loan_status', axis=1),
                                                    df[' loan_status'],
                                                    test_size=0.2,
                                                    random_state=0,stratify=df[' loan_status'])

print(y_train)

pre_process = Pipeline([

    ('MinMaxScaler',
     SklearnTransformerWrapper(MinMaxScaler(), variables = continuous))
])

pre_process.fit(x_train, y_train)

x_train = pd.DataFrame(pre_process.transform(x_train),columns=x_train.columns)
x_test = pd.DataFrame(pre_process.transform(x_test),columns=x_test.columns)
print(y_train)



log_reg = LogisticRegression()
log_param = {

    'C': [0.001,0.01,0.1,1,10,100],
    'penalty': ['l2','none']

}

grid_logreg = GridSearchCV(log_reg, log_param, cv=5, n_jobs=-1,return_train_score=True)
grid_logreg.fit(x_train, y_train)
print(grid_logreg.best_estimator_)

print(f'Best Mean Cross Validation Score is {grid_logreg.best_score_}')
print(f'Best Mean Cross Validation Score is {grid_logreg.best_params_}')
print(f'Train score is {grid_logreg.score(x_train,y_train)}')
print(f'Test score is {grid_logreg.score(x_test,y_test)}')


from sklearn.svm import SVC

ksvc = SVC(probability=True)

ksvc_param = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf', 'sigmoid']
}

ksvc_grid = GridSearchCV(ksvc, ksvc_param, cv=5, return_train_score=True)
ksvc_grid.fit(x_train, y_train)


print(f'Best Mean Cross Validation Score is {ksvc_grid.best_score_}')
print(f'Best Mean Cross Validation Score is {ksvc_grid.best_params_}')
print(f'Train score is {ksvc_grid.score(x_train,y_train)}')
print(f'Test score is {ksvc_grid.score(x_test,y_test)}')

dtree = DecisionTreeClassifier()
dtree_param = {"max_depth": range(1,10),
           "min_samples_split": range(2,10,1),
           "max_leaf_nodes": range(2,6),
              "splitter": ["best", "random"]}
dtree_grid = GridSearchCV(dtree, dtree_param,cv=5, return_train_score=True)
dtree_grid.fit(x_train,y_train)

print(f'Best Mean Cross Validation Score is {dtree_grid.best_score_}')
print(f'Best Mean Cross Validation Score is {dtree_grid.best_params_}')
print(f'Train score is {dtree_grid.score(x_train,y_train)}')
print(f'Test score is {dtree_grid.score(x_test,y_test)}')

ConfusionMatrixDisplay.from_estimator(dtree_grid.best_estimator_,x_test,y_test)
plt.show()

from sklearn.ensemble import StackingClassifier

stack1 = StackingClassifier(estimators=
                            [('log_reg', grid_logreg.best_estimator_),
                             ('kernel_svc',ksvc_grid.best_estimator_),
                             ('dtree',dtree_grid.best_estimator_)],
                            final_estimator=RandomForestClassifier(random_state=0))

stack1_param = {
    'final_estimator__n_estimators': [100,200,300],
    'final_estimator__max_features':['sqrt','log2'],
    'final_estimator__max_depth': range(2,20,2)
}

stack1_grid = GridSearchCV(stack1,stack1_param, cv = 5, n_jobs=-1,return_train_score=True)
stack1_grid.fit(x_train,y_train)

print(f'Best Mean Cross Validation Score is {stack1_grid.best_score_}')
print(f'Best Mean Cross Validation Score is {stack1_grid.best_params_}')
print(f'Train score is {stack1_grid.score(x_train,y_train)}')
print(f'Test score is {stack1_grid.score(x_test,y_test)}')


ConfusionMatrixDisplay.from_estimator(stack1_grid.best_estimator_,x_test,y_test)
plt.show()
