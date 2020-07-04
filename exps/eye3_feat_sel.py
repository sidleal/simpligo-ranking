import numpy
import pandas
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import pickle
import eli5
from eli5.sklearn import PermutationImportance

input_size = 189

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation="relu"))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

pandas.set_option('display.max_colwidth', -1)

# load dataset
df = pandas.read_csv("../data/120sent_eye_features.tsv", delimiter='\t', header=0)

X = df.iloc[:, 3:192]
#Y_tmp = df.iloc[:, 192] #avg_first_pass
#Y_tmp = df.iloc[:, 193] #avg_regression
#Y_tmp = df.iloc[:, 194] #avg_total_pass
#Y_tmp = df.iloc[:, 195] #tot_first_pass
#Y_tmp = df.iloc[:, 196] #tot_regression
Y_tmp = df.iloc[:, 197] #tot_total_pass

#normalizando Y
max_out = 0
for i in range(0,len(Y_tmp)):
    if Y_tmp[i] > max_out:
        max_out = Y_tmp[i]

Y = []
for i in range(0, len(Y_tmp)):
    Y.append(Y_tmp[i]/max_out)

Y = numpy.asanyarray(Y)


# fix random seed for reproducibility
#seed = 7

# evaluate model with standardized dataset
#numpy.random.seed(seed)
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1, verbose=1)

estimators = []
standardScaler = StandardScaler()
estimators.append(('standardize', standardScaler))
estimators.append(('mlp', estimator))
pipeline = Pipeline(estimators)

pipeline.fit(X, Y)
prediction = pipeline.predict(X)

y_test = numpy.asanyarray(Y)

r, p_value = scipy.stats.pearsonr(y_test, prediction)

error = 0
tot_samples = len(prediction)
for j in range(0, tot_samples ):
    print("%s - %s" % (y_test[j], prediction[j]))
    error = error + (prediction[j] - y_test[j]) ** 2

mse = error / len(prediction)
rmse = numpy.math.sqrt(mse)

perm = PermutationImportance(pipeline, random_state=1)
res = perm.fit(X, y_test)
ret = eli5.format_as_dict(eli5.explain_weights(res, top=180, feature_names=X.columns.tolist()))
print(ret)

print("---------------")
print("MEAN SQUARED ERROR:", mse)
print("ROOT MEAN SQUARED ERROR:", rmse)
print("PEARSON'S CORRELATION COEFFICIENT:", r, "p-value", p_value)

print("---------------")

for i in ret['feature_importances']['importances']:
    print(i)

