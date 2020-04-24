'''Example script showing how to use stateful RNNs
to model long sequences efficiently. Adapted from Keras examples.
'''
''' NECCESSARY MODULES BEING IMPORTED '''
from __future__ import print_function
import numpy as np
from loaddata import dataimport,generatedata
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# since we are using stateful rnn tsteps can be set to 1

tsteps = 1
batch_size = 50
epochs = 1000
neurons = 100
layers = 1
period = 100
length = 500

data, y, testdata = dataimport()
print('Input shape:', data.shape)

print('Output shape:', y.shape)

'''CREATING RNN MODEL '''

print('Creating RNN Model...')
model = Sequential()
if layers<2:
    model.add(LSTM(neurons,
                   input_shape=(tsteps, 1),
                   batch_size=batch_size,
                   return_sequences=False,
                   stateful=True))
else:
    model.add(LSTM(neurons,
                   input_shape=(tsteps, 1),
                   batch_size=batch_size,
                   return_sequences=True,
                   stateful=True))
    for i in range(layers-2):
        model.add(LSTM(neurons,
                       return_sequences=True,
                       stateful=True))
    model.add(LSTM(neurons,
                   return_sequences=False,
                   stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

''' TRAINING THE MODEL '''

print('Training')
for i in range(epochs):
    print('Epoch', i, '/', epochs)

 
    model.fit(data, y,batch_size=batch_size,epochs=1,verbose=1,shuffle=False)
    model.reset_states()

''' PREDICTING USING MODEL '''

print('Predicting')
#Invert and shrink the test set!
predicted_output = model.predict(data, batch_size=batch_size)

''' PLOTTING RESULTS '''

print('Plotting Results')
plt.plot(y)
plt.plot(predicted_output)
plt.title('Expected and Predicted\ntsteps = '+str(tsteps)+', batch_size = '+str(batch_size)+', epochs = '+str(epochs)+'\nneurons = '+str(neurons)+', layers = '+str(layers)+', length = '+str(length)+', period = '+str(period))
plt.show()
print('Completed RNN Model...')

traindata, testdata = generatedata(data)

''' CREATING SVR MODEL '''

print('Creating SVR Model...')
svr = SVR(C=1.0, epsilon=0.2)
svr.fit(traindata, y)

svr_pred = svr.predict(traindata)

''' PLOTTING RESULTS '''

print('Plotting Results')
plt.plot(y)
plt.plot(svr_pred)
plt.title('Expected and Predicted for SVR')
plt.show()

svr_pred = svr.predict(testdata)
#print(svr_pred)

print(svr.score(traindata,y))
print('Completed SVR Model...')

''' CREATING RANDOMFOREST MODEL '''

print('RanfomForest Model...')
rfregr = RandomForestRegressor(random_state=0)
rfregr.fit(traindata, y)
rfregr_pred = rfregr.predict(traindata)

''' PLOTTING RESULTS '''

print('Plotting Results')
plt.plot(y)
plt.plot(rfregr_pred)
plt.title('Expected and Predicted for RandomForest')
plt.show()

rfregr_pred = rfregr.predict(testdata)
#print(rfregr_pred)

print(rfregr.score(traindata,y))
print('Completed RandomForest Model...')

''' CREATING GRADIENT BOOSTING MODEL '''

print('GradientBossting Model...')

gbregr = GradientBoostingRegressor(random_state=0)
gbregr.fit(traindata, y)

gbregr_pred = rfregr.predict(traindata)

''' PLOTTING RESULTS '''

print('Plotting Results')
plt.plot(y)
plt.plot(gbregr_pred)
plt.title('Expected and Predicted for GradientBossting')
plt.show()

gbregr_pred = gbregr.predict(testdata)
#print(gbregr_pred)

print(gbregr.score(traindata,y))
print('Completed GradientBosting Model...') 

