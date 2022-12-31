import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



input_test_csv = pd.read_csv('reviews_Video_Games_test.csv')
input_training_csv = pd.read_csv('reviews_Video_Games_training.csv')


train_x = input_training_csv.drop('data', axis=1)
train_y = input_training_csv['data']
test_x = input_test_csv.drop('data', axis=1)
test_y = input_test_csv['data']

dtree = DecisionTreeClassifier(max_depth=2)
dtree.fit(test_x, test_y)



predictions = dtree.predict(test_x)
acc = accuracy_score(test_y, predictions)
print(classification_report(test_y, predictions))