### import packages

# lets basic import packages

from keras.models import Sequential
from keras.layers import Activation, Dense
import tensorflow
import keras.models
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score

# from pandas_profiling import ProfileReport

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost.sklearn import XGBClassifier

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


## import data

df = pd.read_csv('data/dataset.csv')
df = shuffle(df, random_state=42)
df.head()


# remove ('_') underscore in the text

for col in df.columns:
    df[col] = df[col].str.replace('_', ' ')

df.head()

# charactieristics of data

df.describe()

# check null values

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
print(null_checker)

# plot of null value

plt.figure(figsize=(10, 5), dpi=140)
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index,
           rotation=45, horizontalalignment='right')
plt.title('Ratio of Null values')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()

cols = df.columns

data = df[cols].values.flatten()

reshaped = pd.Series(data)
reshaped = reshaped.str.strip()
reshaped = reshaped.values.reshape(df.shape)

df = pd.DataFrame(reshaped, columns=df.columns)
df.head()


# lets fill nan values

df = df.fillna(0)
df.head()

# lets explore symptom severity

df_severity = pd.read_csv(
    '/kaggle/input/disease-symptom-description-dataset/Symptom-severity.csv')
df_severity['Symptom'] = df_severity['Symptom'].str.replace('_', ' ')
df_severity.head(10)

# overall list

df_severity['Symptom'].unique()


# lets encode sysptoms in the data

vals = df.values
symptoms = df_severity['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df_severity[df_severity['Symptom']
                                            == symptoms[i]]['weight'].values[0]


df_processed = pd.DataFrame(vals, columns=cols)
df_processed.head()

# assign symptoms with no rank to zero

df_processed = df_processed.replace('dischromic  patches', 0)
df_processed = df_processed.replace('spotting  urination', 0)
df_processed = df_processed.replace('foul smell of urine', 0)


# split data
data = df_processed.iloc[:, 1:].values
labels = df['Disease'].values


# split trai and test data
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# shape of train data
print(X_train[0])
print(X_train[1])


def performance_evaluator(model, X_test, y_test):
    """
    model: Load the trained model
    X_test: test data
    y_test: Actual value

    """

    y_predicted = model.predict(X_test)

    precision = precision_score(y_test, y_predicted, average='micro')*100

    accuracy = accuracy_score(y_test, y_predicted)*100

    f1 = f1_score(y_test, y_predicted, average='macro')*100

    recall = recall_score(y_test, y_predicted, average='macro')*100

    print('precision----->', precision)
    print('\n************************')
    print('Accuracy----->', accuracy)
    print('\n************************')
    print('F1 Score----->', f1)
    print('\n************************')
    print('Recall----->', recall)
    print('\n************************')
    return accuracy, precision, f1, recall


# plot classification metrix

def confusion_plot(model, X_test, y_test):
    """
    to plot confusion metrix
    """
    plt.figure(figsize=(10, 10), dpi=150)

    y_pred = model.predict(X_test)
    con_me = confusion_matrix(y_test, y_pred)
    sns.heatmap(con_me, annot=True)


# lets play with Support Vector Machine

SVM_init = SVC()
model_SVM_init = SVM_init.fit(X_train, y_train)


_1, _2, _3, _4 = performance_evaluator(model_SVM_init, X_test, y_test)


# support Vector machine Hyperparameter tuned

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
grid.fit(X_train, y_train)


# best estimator

print(grid.best_estimator_)
print(grid.best_params_)

# lets built based SVC model.

hyper_tuned_svc = SVC(C=10, gamma=0.1, kernel='rbf')
hyper_tuned_svc.fit(X_train, y_train)

# lets calculate performance
_1, _2, _3, _4 = performance_evaluator(hyper_tuned_svc, X_test, y_test)


confusion_plot(hyper_tuned_svc, X_test, y_test)

# Gradient Boosting Classifier with out hyperparameter tuning

GBC_model = GradientBoostingClassifier()
GBC_model.fit(X_train, y_train)


# lets calculate performance Gradient Boosting Classifier

_1, _2, _3, _4 = performance_evaluator(GBC_model, X_test, y_test)


confusion_plot(GBC_model, X_test, y_test)


# some meaningfull predictions

GBC_model.predict([X_test[1]])

GBC_model.predict([X_test[2]])

GBC_model.predict([X_test[3]])

# use for loop to predict every test data

for i in range(len(X_test)):
    print(GBC_model.predict([X_test[i]]))
    print("The actual value is: ", y_test[i])

# check the accuracy of the model on the test data manually

# lets check the accuracy of the model on the test data manually

count = 0
for i in range(len(X_test)):
    if GBC_model.predict([X_test[i]]) == y_test[i]:
        count += 1

print("The accuracy of the model is: ", count/len(X_test))

# find wrong predictions

count = 0
for i in range(len(X_test)):
    if GBC_model.predict([X_test[i]]) != y_test[i]:
        print(GBC_model.predict([X_test[i]]))
        print("The actual value is: ", y_test[i])
        count += 1

print("The number of wrong predictions are: ", count)


# use neural network to predict

# lets use neural network to predict


# create model

#sequential is not defined
8


model = Sequential()
model.add(Dense(12, input_dim=132, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(X_train, y_train, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X_test)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

# lets calculate performance of neural network

_1, _2, _3, _4 = performance_evaluator(model, X_test, y_test)
