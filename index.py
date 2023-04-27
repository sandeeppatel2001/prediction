import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
# Define the input and output data
Input = pd.read_csv("training_input.csv")
X = np.array(Input)
# Input_features
Output = pd.read_csv("training_output.csv")
Y = np.array(Output)



print(Y.shape)
# Define the neural network model
model = tf.keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=[8]),
  layers.Dense(64, activation='relu'),
  layers.Dense(7)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.001))

# Train the model
history = model.fit(X, Y, epochs=1000)


x2=np.array(pd.read_csv("test_input.csv"))
y2=np.array(pd.read_csv("testoutput.csv"))
# Evaluate the model
test_loss = model.evaluate(x2, y2)
# Make predictions
predictions = model.predict(x2)

# Print the results
print("Predictions:", predictions)
print("Test loss:", test_loss)
