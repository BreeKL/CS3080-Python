from FileUtils import FileUtils as futils
import os
import matplotlib.pyplot as plt
import numpy as np


#****************************
#open data file
my_file = 'Assignment2\\Regression\\rounded_hours_student_scores.csv'
if os.path.exists(my_file):
    print(f"{my_file} exists:")
else: print(f"{my_file} does not exist")

data = futils.read_csv_file(my_file)

#print(data)


#***************************
#load data
x = []
y = []

x = [float(item["Hours"]) for item in data if isinstance(item, dict) and "Hours" in item]
#print("x = ", x)

y = [float(item["Scores"]) for item in data if isinstance(item, dict) and "Scores" in item]
#print("y = ", y)

x_test  = np.array(x[50::], dtype=float)
x = np.array(x[:50:], dtype=float)
#print("x = \n", x)
#print("x_test = \n", x_test)

y_test  = np.array(y[50::], dtype=float)
y = np.array(y[:50:], dtype=float)
#print("y = \n", y)
#print("y_test = \n", y_test)


#***************************
#calculate means, slope, and y intercept
x_mean = np.mean(x)
y_mean = np.mean(y)
#print("x mean is", x_mean, "and y mean is", y_mean)
 
nominator = np.sum((x - x_mean) * (y - y_mean) )
denominator = np.sum((x - x_mean) ** 2)
#print("nominator is", nominator, "and denominator is", denominator)

m = nominator/denominator
c = (y_mean - m * x_mean)
#print("Slope is", m, "and y intercept is", c)


#***************************
#creates linear prediction lines
def predict(x, m, c):
    return m * x + c

predictions = predict(x, m, c)
#print("Training predictions:", predictions)

test_predictions = predict(x_test, m, c)
#print("Test predictions:", test_predictions)


#***************************
#error function
def mean_squared_error(y_gt, y_pred):
    return np.mean((y_gt - y_pred)**2)

train_mse = mean_squared_error(y, predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"\nTrain Mean Squared Error (y - predictions): {train_mse}")
print(f"\nTest Mean Squared Error (y_test - test_predictions): {test_mse}\n")


#***************************
#logging
train_results = [{'Hours':float(x[i]), 'Score':float(y[i]), 'Predicted Score':float(predictions[i])} for i in range(len(x))]

#print("Training results:", train_results)

for item in train_results:
    futils.write_csv_file('Assignment2\\Regression\\test_predictions.csv', item)

#***************************
# Plotting the results
plt.scatter(x, y,
    color='blue',
    label='Actual data')
plt.scatter(x_test, y_test,
    color='red',
    label='Test data')
plt.plot(x, predictions,
    color='red',
    label='Regression line')
plt.plot(x_test, test_predictions,
    color='black',
    label='Test data')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.legend()
plt.title('Linear Regression From Scratch')
plt.show()
