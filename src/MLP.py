from keras.models import Sequential
from keras.layers import Dense

from analyz import X_train, X_test, y_train_encoded, y_test_encoded


model = Sequential()
model.add(Dense(64, input_dim=23, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train_encoded, epochs=5, batch_size=32)
model.summary()

loss, accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Accuracy: {accuracy * 100}%")

