from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=10)            # checks most similar closest classes 
knn.fit(x_train, y_train)

knn.score(x_train, y_train)
knn_prediction= knn.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn_prediction)

from sklearn.metrics import mean_absolute_error , mean_squared_error

print(" Regression algo absolute error")
print( mean_absolute_error( y_test , linear_predictions))
np.sqrt( mean_squared_error( y_test , linear_predictions))

print(" KNN algo absolute error")
print( mean_absolute_error( y_test , knn_prediction))
np.sqrt( mean_squared_error( y_test , knn_prediction))
