# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 23:28:50 2023

@author: Kushum
"""

# Step 1: Loading Libraries
import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


#Step 2: Ste working directory
path = 'D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Assignment_6\\WD'
os.chdir(path)

#Step 3: Load the imputed time series data
df = pd.read_csv('1.J_17_Imputed.csv')
df = df.set_index('Date')


#Step 4: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(df)


#Step 5: Split the data into input and output sequences
sequence_length = 10
X = []
y = []
for i in range(len(data)-sequence_length-1):
    X.append(data[i:(i+sequence_length), 0])
    y.append(data[i+sequence_length, 0])
X = np.array(X)
y = np.array(y)

#Step 6: Split the data into training and testing sets
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#Step 7: Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Step 8: Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

#Step 9: Model fitting
#Step 9.1: Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1, validation_data=(X_test, y_test))

#Step 9.2: Forecasting using the model
train_forecast = model.predict(X_train)
test_forecast = model.predict(X_test)

#Step 10: Invert the predictions to their original scale
train_forecast = scaler.inverse_transform(train_forecast)
y_train = scaler.inverse_transform([y_train])
test_forecast = scaler.inverse_transform(test_forecast )
y_test = scaler.inverse_transform([y_test])

#Step 11: Evaluate the model
#Step 11.1: Plot
# Step 11.1.1: Create DataFrame for train dataset (by creating a dictionary with the two variables)
data = {'y_train': y_train[0], 'train_forecast': train_forecast[:,0]}
df_train = pd.DataFrame(data)      # Create a dataframe from the dictionary
print(df_train)                    # Print the dataframe
#Step 11.1.2: plot for observed vs Forcasted train dataset
plt.plot(df_train['y_train'],df_train['train_forecast'], 'ro', markersize=0.25)
plt.xlabel('Observed_train')
plt.ylabel('Forecast_train')

#Step 11.1.3: Time series plot
#Create a new figure and plot the actual data as a blue line
fig = plt.figure(figsize=(8, 6))
plt.plot(df_train['y_train'], color='blue', label='Actual')
# Plot the predicted data as a red line
plt.plot(train_forecast, color='red', label='Forecast')
# Add axis labels, a title, and a legend
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Forecast Time Series Data')
plt.legend()


# Step 11.1.4: Create DataFrame for test dataset (by creating a dictionary with the two variables)
data = {'y_test': y_test[0], 'test_forecast': test_forecast [:,0]}
df_test = pd.DataFrame(data)      # Create a dataframe from the dictionary
print(df_test)                    # Print the dataframe
#Plot Observed vs Forecasted test dataset
plt.plot(df_test['y_test'],df_test['test_forecast'], 'bo', markersize=0.25)
plt.xlabel('Observed_test')
plt.ylabel('Forecast_test')

#Step 11.2: Calculate MSE between y_train and train_predict
train_score = np.sqrt(mean_squared_error(y_train[0], train_forecast[:,0]))
test_score = np.sqrt(mean_squared_error(y_test[0], test_forecast [:,0]))
print('Train RMSE:', train_score)
print('Test RMSE:', test_score)


