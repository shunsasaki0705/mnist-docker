import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# define model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(x_train, y_train, epochs=5)

# evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print('\nTest accuracy:', test_acc)

# save the model
model.save('mnist_model.h5')

# GPU usage
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
