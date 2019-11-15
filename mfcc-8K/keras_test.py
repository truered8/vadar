import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(1, kernel_size=(1,1), activation='sigmoid', input_shape=(1, 2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='MSE',
			  optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
			  metrics=['accuracy'])
model.fit(np.array([[[[1, 1],[1, 1]]]]), np.array([1]), batch_size=1, epochs=1, verbose=1)
model.save('conv_test.h5')
