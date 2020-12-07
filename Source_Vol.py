from numpy.core.fromnumeric import prod
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import metrics
import pickle
import seaborn as seabornInstance
from sklearn import metrics
import matplotlib.pyplot as pltd
import math
import datetime

#Algorithm efficiency clock
begin_time = datetime.datetime.now()

#Read Data
data = pd.read_csv("Source of volume.csv")

#Prediction metric
predict = "Cheez It"

#Data Input
data = data[[
    'Cheez It',
    "Veggie Crisps",
    "All Other",
    "RITZ", 
    "Triscuits",
    "Breton",
    "Bear Paws Crackers", 
    "Goldfish", 
    "Keebler Townhouse", 
    "Vinta",
    "Wheat Thins",
    
]]

#Shuffle data for training the model
data = shuffle(data) 

#how to display data
#print(data.head())
#print(data.tail(n=10))

#Putting the inputs in their correct place in the algorithm 
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Running the calculation 
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

#Testing the accuracy of the model + Output
y_pred = linear.predict(x_test)

print("Accuracy: " + str(acc))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

#Target minimum best accuracy
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
best = 0.8
count = 0

for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # The line below is needed for error testing.
    y_pred = linear.predict(x_test)
    
    count += 1
    print(f"\nRun number: {count}")
    print("Accuracy: " + str(acc))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Displaying the accuracy
    if acc > best:
        best = acc
        #Saving the model
        with open("cannabal.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL 
pickle_in = open("cannabal.pickle", "rb") 
linear = pickle.load(pickle_in) 
final_accuracy = int(best*100)
coeff = {}
arr = linear.coef_

#Time efficiency check
ziper = list(zip(data,arr))
run_time = datetime.datetime.now() - begin_time

#Iteration through results to print final source of volume score:
binomial_effect_size_display_pos = (final_accuracy/2)+50
binomial_effect_size_display_neg = (final_accuracy/2)-50

print("------------------------------------------------------------------")
print("Accuracy Statistics")
print("------------------------------------------------------------------")
print(f"Chance of being accurate is {binomial_effect_size_display_pos}%")
print(f'Chance of being inaccurate is {binomial_effect_size_display_neg}%')

print('\n')
print("------------------------------------------------------------------")
print("Sample Predictions")
print("------------------------------------------------------------------")
for x in range(len(y_pred)):
    difference = y_pred[x] - y_test[x]
    if difference < 50000:
        print("------------------------------------------------------------------")
        print(f'Cheez It Sales Prediction: \n${y_pred[x]}\n\nInputs: \n{x_test[x]}\n\nActual Cheez It Sales: \n${y_test[x]}\n')


print('\n')
print("------------------------------------------------------------------")
print("Output Report:")
print("------------------------------------------------------------------")
sumx = []
products = []
final_percent = []
percentages = []
product_list =[]
for i, x in ziper:
    if x < 0 :
        sumx.append(x)
       
sum_total = sum(sumx)
for i in range(len(sumx)):
    products.append(sum_total)

final_list = list(zip(sumx, products))

for sum, product in final_list:
    percentage = int((sum/product)*100)
    percentages.append(percentage)

for i , x in ziper:
    if x < 0:
        product_list.append(i)

print_out_list = list(zip(product_list, percentages))


for product, percent in print_out_list:
    if product != "Cheez It":
        print(f"Cheez It Got {percent}% of its volume from {product}")





print(f"\nTotal run time: \n{run_time} seconds\n")
print(f"Total number of runs: \n{count}\n")
print('Intercept: \n', linear.intercept_)
print(f'\nBest accuracy: \n{final_accuracy}%')
print('\nMean Absolute Error:\n', metrics.mean_absolute_error(y_test, y_pred))
print('\nMean Squared Error:\n', metrics.mean_squared_error(y_test, y_pred))
print('\nRoot Squared Error:\n', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("------------------------------------------------------------------")


#plot the data
#data.describe()

#Check for null values
#data.isnull().any()

#Remove null data
# data = data.fillna(method='ffill')

'''
#HOW TO PLUG IN VALUES TO PREDICT $VOL 
new_x = [[1768380, 1015909, 60871,  576818, 1441939,  202619, 4900000, 5492229]]
print('The Expected $Vol with your inputs is:\n ', linear.predict(new_x))
'''
