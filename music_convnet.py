from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adamax
from keras.utils import to_categorical
from sklearn import model_selection
from PIL import Image
import numpy as np
import os

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def evaluate(predictions, labels):
    p = list(map(np.argmax, predictions))
    l = list(map(np.argmax, labels))
    accuracy = sum(list(map(lambda x, y: 1 if x == y else 0, p, l))) / len(l)
    confusion_matrix = []
    for i in range(5):
        confusion_matrix.append([])
        for j in range(5):
            confusion_matrix[-1].append(sum(list(map(lambda y, x: 1 if x == i and y == j else 0, p, l))))
    return accuracy, confusion_matrix


genres = ["Country", "EDM", "HipHop", "Latin", "Rock"]

images = []
for genre in genres:
    for i in range(10):
        filename = "./" + genre + "/" + str(i) + ".png"
        images.append(load_image(filename)[np.newaxis, :, 5 * 128:6 * 128])


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))

adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

classifier.compile(optimizer = adamax, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_struct = classifier.to_json()
fmod_struct = open(os.path.join("./model", "cnn_model.json"), "wb")
fmod_struct.write(model_struct.encode())
fmod_struct.close()

data = np.concatenate(images, axis = 0)[...,np.newaxis]
print(data.shape)
labels = to_categorical(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]))
training_accuracies = []
validation_accuracies = []

kf = model_selection.KFold(n_splits=5)

for train_index, validation_index in kf.split(data, y = labels):
    classifier.fit(x = data[train_index], y = labels[train_index], batch_size = 32, epochs = 10, validation_data = (data[validation_index], labels[validation_index]))

    cnn_train_predictions = classifier.predict(data[train_index]).tolist()

    accuracy, confusion_matrix = evaluate(cnn_train_predictions, labels[train_index].tolist())
    training_accuracies.append(accuracy)
    print("Training accuracy:", accuracy)
    print("Confusion matrix:", confusion_matrix)

    cnn_validation_predictions = classifier.predict(data[validation_index])

    accuracy, confusion_matrix = evaluate(cnn_validation_predictions, labels[validation_index].tolist())
    validation_accuracies.append(accuracy)
    print("Validation accuracy:", accuracy)
    print("Confusion matrix:", confusion_matrix)

print("Average training accuracy:", sum(training_accuracies) / len(training_accuracies))
print("Average validation accuracy:", sum(validation_accuracies) / len(validation_accuracies))