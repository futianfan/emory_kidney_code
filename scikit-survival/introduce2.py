import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from time import time 

dataFolder = '../data'
trainFile = os.path.join(dataFolder, 'train_cox.npy')
trainData = np.load(trainFile)
testFile = os.path.join(dataFolder, 'test_cox.npy')
testData = np.load(testFile)
print(trainData.shape)
lowerbound = 80
upperbound = 300
#trainData = trainData[:,:cutoff]
#testData = testData[:,:cutoff]
train_num = trainData.shape[0]
trainData = np.concatenate([trainData, testData], 0)
trainData = trainData[:,lowerbound:upperbound]

data_y = trainData[:,:2] 
data_x = trainData[:,2:]
x, y = data_x.shape 
data_x += 0.001 * np.random.random((x,y))
gf_day = list(trainData[:,0])
gf_1year_label = list(trainData[:,1])
gf_1year_label = list(map(lambda x:x==1, gf_1year_label))
dt=np.dtype('bool,float')
data_y = [(gf_1year_label[i], gf_day[i]) for i in range(len(gf_1year_label))]
data_y = np.array(data_y, dtype=dt)

t1 = time()
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x[:train_num], data_y[:train_num])
print('fitting estimate cost {} seconds'.format(int(time() - t1)))
print(estimator.score(data_x[train_num:], data_y[train_num:]))



'''
data_x, data_y = load_veterans_lung_cancer()

#pd.DataFrame.from_records(data_y[[11, 5, 32, 13, 23]], index=range(1, 6))


time, survival_prob = kaplan_meier_estimator(data_y["Status"], data_y["Survival_in_days"])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")

print(data_x["Treatment"].value_counts())

for treatment_type in ("standard", "test"):
    mask_treat = data_x["Treatment"] == treatment_type
    time_treatment, survival_prob_treatment = kaplan_meier_estimator(
        data_y["Status"][mask_treat],
        data_y["Survival_in_days"][mask_treat])
    
    plt.step(time_treatment, survival_prob_treatment, where="post",
             label="Treatment = %s" % treatment_type)

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")

for value in data_x["Celltype"].unique():
    mask = data_x["Celltype"] == value
    time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["Status"][mask],
                                                           data_y["Survival_in_days"][mask])
    plt.step(time_cell, survival_prob_cell, where="post",
             label="%s (n = %d)" % (value, mask.sum()))

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")


data_x_numeric = OneHotEncoder().fit_transform(data_x)
data_x_numeric.head()


estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, data_y)
pd.Series(estimator.coef_, index=data_x_numeric.columns)

x_new = pd.DataFrame.from_dict({
    1: [65, 0, 0, 1, 60, 1, 0, 1],
    2: [65, 0, 0, 1, 60, 1, 0, 0],
    3: [65, 0, 1, 0, 60, 1, 0, 0],
    4: [65, 0, 1, 0, 60, 1, 0, 1]},
     columns=data_x_numeric.columns, orient='index')

pred_surv = estimator.predict_survival_function(x_new)
for i, c in enumerate(pred_surv):
    plt.step(c.x, c.y, where="post", label="Sample %d" % (i + 1))
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")




prediction = estimator.predict(data_x_numeric)
result = concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], prediction)
print(result[0])
print(estimator.score(data_x_numeric, data_y))


def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

scores = fit_and_score_features(data_x_numeric.values, data_y)
pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False)


pipe = Pipeline([('encode', OneHotEncoder()),
                 ('select', SelectKBest(fit_and_score_features, k=3)),
                 ('model', CoxPHSurvivalAnalysis())])




param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
gcv = GridSearchCV(pipe, param_grid, return_train_score=True)
gcv.fit(data_x, data_y)

pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)

pipe.set_params(**gcv.best_params_)
pipe.fit(data_x, data_y)

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])



'''













