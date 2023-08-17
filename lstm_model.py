#import libraries
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import statistics as st
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#load train data
train_data= pd.read_csv('HDFC1.csv')
#drop unnecessary fraatures
selected_train_data=train_data.drop(['Symbol','Series','High','Low','Turnover','%Deliverble','Trades','Open','VWAP','Last','Deliverable Volume','Prev Close'],axis=1)


#calculation of 7 days moving average
def MA_CALC(a):
    v=[]
    for i in range(a.shape[0]):
        if i>=6:
            ll=[n for n in a['Close'][i-6:i+1]]
            v.append([st.mean(ll)])
        else:
            v.append([a['Close'][i]])
    v=pd.DataFrame(v,columns=['7_MA'])   
    d=a.join(v)
    return d
selected_train_data2=MA_CALC(selected_train_data)


#normalise
scalar=MinMaxScaler(feature_range=(0,1))
def data_norm(d,scalar):
    time=pd.DataFrame(d['Date'])
   

    sca_Close=pd.DataFrame(scalar.fit_transform(d['Close'].values.reshape(-1,1)),columns=['Close'])
    SCALED_DATA=time.join(sca_Close)
    sca_Volume=pd.DataFrame(scalar.fit_transform(d['Volume'].values.reshape(-1,1)),columns=['Volume'])
    SCALED_DATA=SCALED_DATA.join(sca_Volume)
    sca_7_MA=pd.DataFrame(scalar.fit_transform(d['7_MA'].values.reshape(-1,1)),columns=['7_MA'])
    SCALED_DATA=SCALED_DATA.join(sca_7_MA)
    return SCALED_DATA
SCALED_DATA=data_norm(selected_train_data2,scalar)


#formation of x_train and y_train
def feature_matrix(SCALED_DATA):
    predict_days=60                                                 #we will predict closing prices based on featurres from past 60 days
    x_train=[]
    y_train=[]
    for i in range(predict_days,SCALED_DATA.shape[0]):
        x_temp=[]
        x_temp.extend(SCALED_DATA['Close'][i-predict_days:i])       #collecting closing prices for past 60 days for each target  
        x_temp.extend(SCALED_DATA['Volume'][i-predict_days:i])      #collecting volumes for past 60 days for each target
        x_temp.extend(SCALED_DATA['7_MA'][i-predict_days:i])        #collecting 7 days moving average for past 60 days for each target
        y_train.append(SCALED_DATA['7_MA'][i])
        x_train.append(x_temp)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1])) 
    return x_train,y_train
x_train,y_train=feature_matrix(SCALED_DATA)


#build model
def LSTM_model():
    
    model = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model
model = LSTM_model()                                            #instance of LSTM model
model.summary()
model.compile(optimizer='adam', 
              loss='mean_squared_error')
Model: "sequential"


# Define callbacks
checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf', verbose = 2,save_best_only = True)
model.fit(x_train, y_train,epochs=25, batch_size = 32,callbacks = [checkpointer])



#load test data
data2= pd.read_csv('HDFC2.csv')
#preprocess and form x_test and y_test
test_data=data2.drop(['Symbol','Series','High','Low','Turnover','%Deliverble','Trades','Open','VWAP','Last','Deliverable Volume','Prev Close'],axis=1)
test_data2=MA_CALC(test_data)
norm_test_data=data_norm(test_data2,scalar)
x_test,y_test=feature_matrix(norm_test_data)

#prediction of closing prices
predicted_prices = model.predict(x_test)

#eliminating negative predicted prices if any
m=[]
for i in range(predicted_prices.shape[0]):
    if predicted_prices[i]<0:
        predicted_prices[i]=0.02
    else :
        m.append(1)
#scaling normalised predicted prices back to proper scale        
predicted_prices=scalar.inverse_transform(predicted_prices)        
#combined data of x_train and x_test
total_data=pd.read_csv('HDFC_total.csv')


#plotting predicted prices and actual prices (start date: 20/04/2018)
plt.figure(figsize=(18,12))
plt.plot(predicted_prices,label=2)
plt.plot(list(total_data['Close'][4559:]),label=1)  
plt.legend()
plt.annotate('start: 20/04/2018', xy =(350, 1380),xytext =(350, 1380))
plt.legend()
plt.annotate('INFY', xy =(375, 1400),xytext =(375, 1400))
                
#print list of predicted prices and actual prices                
predicted_prices
total_data['Close'][4559:]


Y_Test = np.reshape(list(total_data['Close'][4558:]),(748,1))

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print("Mean Squared Error:", mean_squared_error(Y_Test, predicted_prices))
print("Mean Absolute Error:", mean_absolute_error(Y_Test, predicted_prices))
print("Coefficient of Determination:", r2_score(Y_Test, predicted_prices))




