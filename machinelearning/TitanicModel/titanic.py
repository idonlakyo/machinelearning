import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

trainFilePath = 'C:\\Users\\Daniel\\Downloads\\train.csv'
testFilePath = 'C:\\Users\\Daniel\\Downloads\\test.csv'

data = pd.read_csv(trainFilePath)
testData = pd.read_csv(testFilePath)


y = data.Survived

vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

x = data[vars]
testx = testData[vars]

for key, value in x.iteritems():
    if key == 'Sex':
        for i in range(value.size):
            if value[i] == 'male':
                x.at[i, key] = 1
            else:
                x.at[i, key] = 2

    if key == 'Age':
        for i in range(value.size):
            if math.isnan(value[i]):
                x.at[i, key] = 35

    if key == 'Embarked':
        for i in range(value.size):
            if value[i] == 'C':
                x.at[i, key] = 1
            elif value[i] == 'S':
                x.at[i, key] = 2
            elif value[i] == 'Q':
                x.at[i, key] = 3
            elif math.isnan(value[i]):
                x.at[i, key] = 2

    if key == 'Fare':
        for i in range(value.size):
            if math.isnan(value[i]):
                x.at[i, key] = 10

    if key == 'Cabin':
        for i in range(value.size):
            if type(value[i]) == type(''):
                x.at[i, key] = 1
            else:
                x.at[i, key] = 2

for key, value in testx.iteritems():
    if key == 'Sex':
        for i in range(value.size):
            if value[i] == 'male':
                testx.at[i, key] = 1
            else:
                testx.at[i, key] = 2

    if key == 'Age':
        for i in range(value.size):
            if math.isnan(value[i]):
                testx.at[i, key] = 35

    if key == 'Embarked':
        for i in range(value.size):
            if value[i] == 'C':
                testx.at[i, key] = 1
            elif value[i] == 'S':
                testx.at[i, key] = 2
            elif value[i] == 'Q':
                testx.at[i, key] = 3
            elif math.isnan(value[i]):
                testx.at[i, key] = 2

    if key == 'Fare':
        for i in range(value.size):
            if math.isnan(value[i]):
                testx.at[i, key] = 10

    if key == 'Cabin':
        for i in range(value.size):
            if type(value[i]) == type(''):
                testx.at[i, key] = 1
            else:
                testx.at[i, key] = 2

trainx, valx, trainy, valy = train_test_split(x, y, random_state=1)

model = DecisionTreeRegressor(random_state=1)
model.fit(trainx, trainy)

predictions = model.predict(valx)

mae = mean_absolute_error(predictions, valy)
print(mae)
testPrediction = model.predict(testx)
predict = []
for i in range(testPrediction.size):
    predict.append(int(round(testPrediction[i])))

output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predict})
output.to_csv('submission.csv', index=False)





