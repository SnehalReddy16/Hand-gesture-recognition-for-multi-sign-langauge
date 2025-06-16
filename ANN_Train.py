import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('data.pickle', 'rb'))

#print(len(data_dict['data']))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# print(type(data[0]))
# print(type(labels[0]))
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Define ANN model
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(84,)),  # Input layer with 84 units (assuming each list has 84 elements)
    Dense(units=32, activation='relu'),  # Hidden layer with 32 units
    Dense(units=len(set(labels)), activation='softmax')  # Output layer with number of classes (unique labels)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32)  # Adjust epochs and batch size as needed

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, np.argmax(y_predict, axis=1))  # Use argmax for predicted class index

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
model.save('model_new_26032025.h5')

