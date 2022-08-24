#Import needed tools
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from DataProcessing import DataProcessor

#Create class for data processing
dp = DataProcessor()
x_train, x_test, y_train, y_test = dp.load_data(test_size = 0.25)

#Create model by MLPClassifier
model = MLPClassifier(alpha = 0.01,
                      batch_size = 256,
                      epsilon = 1e-08,
                      hidden_layer_sizes = (300, ),
                      learning_rate = 'adaptive',
                      max_iter = 1000)
#Fit the model
model.fit(x_train, y_train)

#Get the predict for the test set
y_predict = model.predict(x_test)

#Calculate the accuracy of the model
accuracy = accuracy_score(y_true = y_test, y_pred = y_predict)
print("\nAccuracy of the model: {:.2f}%".format(accuracy*100))
