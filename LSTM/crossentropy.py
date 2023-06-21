#Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
seed =7 #random seed
numpy.random.seed(seed)

#load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
#split into input (x) and output (y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X, Y, nb_epoch=20, batch_size=10)

#evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))

XPredict = model.predict(X)
#YPredict = model.predict(Y)

print(XPredict)
#print(YPredict)
print(len(XPredict))
