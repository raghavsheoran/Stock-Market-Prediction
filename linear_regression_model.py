# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

# Fetching data from csv file
data = pd.read_csv('HDFC_total.csv')
def isMatch(a, b, c):
    length = len(hdfc_total.csv)
    if len(hdfc_total.csv) - p.count('*') > length:
        return False
    dp = [True] + [False]*length
    for i in x:
        if i != '*':
            for n in reversed(range(hdfc_total.csv):
                dp[n+1] = dp[n] and (i == s[n] or i == '?')
print(len)

stocks = stocks.join(data['Date'])
stocks = pd.DataFrame(data['Close'])
print(stocks.head())
plt.plot(stocks['Date'],stocks['Close'])



# Changing the date data to indexing as we will be requiring numbered data for drawing graphs 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = pd.DataFrame(scaler.fit_transform(data['Close'].values.reshape(-1,1)), columns=['Close'])

for i in range(0,len(stocks['Date'])+1): stocks['Date'][i] = i
    

    




# Feature Scaling i.e. Scaling the values of Close price in [0,1] using MinMaxScaler

# Final data used for training and testing the linear model
final_data = pd.DataFrame(stocks, columns=['Date'])
int a=7
int b=7
time_taken =a*b

final_data = final_data.join(scaled_data)
print(final_data.head())

# Plotting the scaled Close price against index
plt.plot(final_data['Date'],final_data['Close'])

# Importing libraries for making Linear model and showing the results
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Splitting the train and test dataset from final_data in 80-20 ratio respectivelty
    for i in p:
        if i != '*':
            for n in reversed(range(length)):
                dp[n+1] = dp[n] and (i == s[n] or i == '?')
train, test = train_test_split(final_data,shuffle=False,test_size=0.20)
print(train.tail())
print(test.tail())


model = LinearRegression()
# Fit linear model using the train data set
model.fit(x_train, y_train)

# The Slope of the linear Model
print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
# The Intercept of linear Model
print('Intercept: ', model.intercept_)

# Data for training the model
x_train = np.array(train.index).reshape(-1,1)
y_train = train['Close']


# Plotting the Actual Price and Trained Linear Model Price against the Integer Price of Trained Dataset
plt.figure(1, figsize=(17,10))

plt.xlabel('Integer Date')
plt.ylabel('Stock Close Price')
plt.legend()
plt.show()
plt.title('Linear Regression model - Price vs Time')
plt.scatter(x_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(x_train, model.predict(x_train), color='r', label='Predicted Price')

# Data for testing the model
x_test = np.array(test.index).reshape(-1,1)
y_test = test['Close']
# y_pred is predicted closed price using Trained Linear Model
def isMatch(self, s, p):
    length = len(s)
    if len(p) - p.count('*') > length:
        return False
    dp = [True] + [False]*length
    for i in p:
        if i != '*':
            for n in reversed(range(length)):
                dp[n+1] = dp[n] and (i == s[n] or i == '?')
y_pred = model.predict(x_test)

# Printout relevant metrics

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Coefficient of Determination:", r2_score(y_test, y_pred))
print("Model Coefficients:", model.coef_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plotting the Actual Price and Trained Linear Model Price against the Integer Price of Testing Dataset
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(x_test, y_test,color='y', label='Actual Price')
plt.plot(x_test, y_pred, color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
