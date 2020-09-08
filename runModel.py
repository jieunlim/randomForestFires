import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from pprint import pprint

INPUT_FILE = "bakedData/fireWeatherData.csv"

fireWeatherData = pd.read_csv(INPUT_FILE, dtype={'fips': object})

# fires are rare but super important, so we should weight
# samples with fire much more heavily
baseWeight = 100
weightMask = fireWeatherData['hasFire']

# set up our weights. create a series of ones, then, where we
# have fires, multiply it by the number of fires and the baseWeight
fireWeatherData['sampleWeight'] = pd.Series(np.ones(len(fireWeatherData)))
fireWeatherData.loc[weightMask, 'sampleWeight'] = fireWeatherData.loc[weightMask, 'activeFires'] * baseWeight

# drop any row with bad data
fireWeatherData.dropna(inplace=True)

# drop fire related or irrelevant data from our inputs
data = fireWeatherData.drop(['fireStarted',
                             'activeFires',
                             'hasFire',
                             'county',
                             'date',
                             'sampleWeight'], axis=1).copy()
# cast fips to an int so the random forest can interpret it
data = data.astype({'fips': int})
# hasFire is our target, but get sampleWeight for later separation
target = fireWeatherData[['hasFire', 'sampleWeight']]

x_train, x_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=2)

sampleWeights = y_train['sampleWeight']

y_train = y_train.drop('sampleWeight', axis=1)
y_test = y_test.drop('sampleWeight', axis=1)

# create and train the model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train, y_train, sample_weight=sampleWeights)

rfcPredict = rfc.predict(x_test)

# gather and print data on the performance of the model
rfcProbs = rfc.predict_proba(x_test)
rfcProbsDF = pd.DataFrame(rfcProbs)
print(rfcProbsDF.describe())
((TN,FP),(FN,TP)) = cfm = confusion_matrix(y_test, rfcPredict)

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
PPV = TP / (TP + FP)
NPV = TN / (TN + FN)
print(f"confusion matrix:\n{cfm}")
print(f"sensitivity: {sensitivity}, specificity: {specificity}")
print(f"PPV: {PPV}, NPV: {NPV}")

print(accuracy_score(y_test, rfcPredict))
