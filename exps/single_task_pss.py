from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import pandas
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Input
from sklearn.model_selection import KFold


input_size = 378 #189*2

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

pandas.set_option('display.max_colwidth', -1)

# load dataset
df = pandas.read_csv("../pss2_features_pairs_align2.tsv", delimiter='\t', header=0)

X = pandas.concat([df.iloc[:, 5:194], df.iloc[:, 195:384]], axis=1)
Y = df.iloc[:, 3]

# evaluate model with standardized dataset
#numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, epochs=30, batch_size=10, verbose=0)

estimators = []
standardScaler = StandardScaler()
estimators.append(('standardize', standardScaler))
estimators.append(('mlp', estimator))
pipeline = Pipeline(estimators)

total_acuracia = 0
total_fscore = 0


n_split = 10
for train_index, test_index in KFold(n_splits=n_split, shuffle=True).split(X):
    X_train, y_train = X.iloc[train_index], Y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], Y.iloc[test_index]

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    y_test = np.asanyarray(y_test)

    for j in range(0, len(prediction)):
        if prediction[j] > 0.5:
            prediction[j] = 1
        else:
            prediction[j] = 0

    print(accuracy_score(y_test, prediction))

    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for j in range(0, len(prediction)):
        if y_test[j] == 1 and prediction[j] == 1:
            truePos+=1
        if y_test[j] == 0 and prediction[j] == 0:
            trueNeg+=1
        if y_test[j] == 1 and prediction[j] == 0:
            falseNeg+=1
        if y_test[j] == 0 and prediction[j] == 1:
            falsePos+=1

    acuracia = (truePos + trueNeg) / (truePos+trueNeg+falsePos+falseNeg)
    print("accuracy %s" % acuracia)
    total_acuracia += acuracia

    recall = truePos / (truePos+falseNeg)
    precision = truePos / (truePos+falsePos)
    specificity = trueNeg / (trueNeg+falsePos)

    fscore = (2 * (precision * recall)) / (precision + recall)

    print("---------------")
    print("recall %s" % recall)
    print("precision %s" % precision)
    print("specificity %s" % specificity)
    print("fscore %s" %fscore)
    total_fscore+= fscore
    print("---------------")

print("=================")
print("total accuracy %s" % (total_acuracia/n_split))
print("total fscore %s" % (total_fscore/n_split))
print("=================")


