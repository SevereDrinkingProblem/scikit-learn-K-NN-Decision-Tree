import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report



def load_data_from_csv(input_csv):
    df = pd.read_csv(input_csv, header = 0)
    csv_headings = list(df.columns.values)
    feature_names = csv_headings[:len(csv_headings) - 1]
    df = df._get_numeric_data()
    numpy_array = df.to_numpy()
    number_of_rows, number_of_columns = numpy_array.shape
    instances = numpy_array[:, 0:number_of_columns - 1]
    labels = []
    for label in numpy_array[:, number_of_columns - 1:number_of_columns].tolist():
        labels.append(label[0])
    return feature_names, instances, labels

input_test_csv = 'reviews_Video_Games_test.csv'
input_training_csv = 'reviews_Video_Games_training.csv'

training_feature_names, training_instances, training_labels = load_data_from_csv(input_csv = input_training_csv)
test_feature_names, test_instances, test_labels = load_data_from_csv(input_csv = input_test_csv)

classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(training_instances, training_labels)
predicted_test_labels = classifier.predict(test_instances)

print(classification_report(test_labels, predicted_test_labels, digits = 3))
