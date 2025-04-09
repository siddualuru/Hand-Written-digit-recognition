import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# 1. Load the MNIST dataset
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# 2. Normalize the pixel values
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# 3. Build the neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128))
model.add(tf.keras.layers.Activation('relu')) # Added Activation layer
model.add(tf.keras.layers.Dense(units=128))
model.add(tf.keras.layers.Activation('relu')) # Added Activation layer
model.add(tf.keras.layers.Dense(units=10))
model.add(tf.keras.layers.Activation('softmax')) # Added Activation layer

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
model.fit(x_train, y_train, epochs=3)

# 6. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Test loss: {loss}")

for x in range (1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("-----------------")
    print("The predicted output is : ",np.argmax(prediction))
    print("-----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()