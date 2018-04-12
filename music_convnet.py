from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adamax
from keras.utils import to_categorical
from sklearn import model_selection
from sklearn.utils import shuffle
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


# images = []
# labels = []

# directory = os.fsencode("./Country")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image = load_image("./Country/" + filename)[np.newaxis, :, :]
#         # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
#         if image.shape != (1, 128, 128):
#             continue
#         images.append(image)
#         labels.append(0)

# directory = os.fsencode("./EDM")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image = load_image("./EDM/" + filename)[np.newaxis, :, :]
#         # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
#         if image.shape != (1, 128, 128):
#             continue
#         images.append(image)
#         labels.append(1)

# directory = os.fsencode("./HipHop")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image = load_image("./HipHop/" + filename)[np.newaxis, :, :]
#         # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
#         if image.shape != (1, 128, 128):
#             continue
#         images.append(image)
#         labels.append(2)

# directory = os.fsencode("./Latin")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image = load_image("./Latin/" + filename)[np.newaxis, :, :]
#         # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
#         if image.shape != (1, 128, 128):
#             continue
#         images.append(image)
#         labels.append(3)

# directory = os.fsencode("./Rock")
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith(".png"):
#         image = load_image("./Rock/" + filename)[np.newaxis, :, :]
#         # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
#         if image.shape != (1, 128, 128):
#             continue
#         images.append(image)
#         labels.append(4)

classifier = Sequential()

classifier.add(Conv2D(64, (2, 2), strides = 2, input_shape = (128, 128, 1), activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(128, (2, 2), strides = 2, input_shape = (64, 64, 64), activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(256, (2, 2), strides = 2, input_shape = (32, 32, 128), activation = 'relu'))
# classifier.add(Dropout(0.2))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(512, (2, 2), strides = 2, input_shape = (16, 16, 256), activation = 'relu'))
# classifier.add(Dropout(0.2))
# classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 5, activation = 'softmax'))

adamax = Adamax(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

classifier.compile(optimizer = adamax, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_struct = classifier.to_json()
fmod_struct = open(os.path.join("./model", "cnn_model.json"), "wb")
fmod_struct.write(model_struct.encode())
fmod_struct.close()

# data = np.concatenate(images, axis = 0)[...,np.newaxis]
# labels = to_categorical(np.array(labels))
# data, labels = shuffle(data, labels)
# # shuffle(data, labels)
# # shuffle(data, labels)
# training_accuracies = []
# validation_accuracies = []

# kf = model_selection.KFold(n_splits=8)

# classifier.save_weights('model.h5')

# cur_best_val = 0
# for train_index, validation_index in kf.split(data, y = labels):
#     classifier.load_weights('model.h5')
#     classifier.fit(x = data[train_index], y = labels[train_index], batch_size = 32, epochs = 50, validation_data = (data[validation_index], labels[validation_index]))

#     cnn_train_predictions = classifier.predict(data[train_index]).tolist()

#     accuracy, confusion_matrix = evaluate(cnn_train_predictions, labels[train_index].tolist())
#     training_accuracies.append(accuracy)
#     print("Training accuracy:", accuracy)
#     print("Confusion matrix:", confusion_matrix)

#     cnn_validation_predictions = classifier.predict(data[validation_index])

#     accuracy, confusion_matrix = evaluate(cnn_validation_predictions, labels[validation_index].tolist())
#     validation_accuracies.append(accuracy)
#     print("Validation accuracy:", accuracy)
#     print("Confusion matrix:", confusion_matrix)
#     if accuracy > cur_best_val:
#         cur_best_val = accuracy
#         classifier.save_weights('trained_model.h5')

# print("Average training accuracy:", sum(training_accuracies) / len(training_accuracies))
# print("Average validation accuracy:", sum(validation_accuracies) / len(validation_accuracies))

# TESTING
print("Testing...")

classifier.load_weights('trained_model.h5')

images = []
labels = []

directory = os.fsencode("./Country_Test")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = load_image("./Country_Test/" + filename)[np.newaxis, :, :]
        # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
        if image.shape != (1, 128, 128):
            continue
        images.append(image)
        labels.append(0)

directory = os.fsencode("./EDM_Test")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = load_image("./EDM_Test/" + filename)[np.newaxis, :, :]
        # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
        if image.shape != (1, 128, 128):
            continue
        images.append(image)
        labels.append(1)

directory = os.fsencode("./HipHop_Test")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = load_image("./HipHop_Test/" + filename)[np.newaxis, :, :]
        # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
        if image.shape != (1, 128, 128):
            continue
        images.append(image)
        labels.append(2)

directory = os.fsencode("./Latin_Test")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = load_image("./Latin_Test/" + filename)[np.newaxis, :, :]
        # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
        if image.shape != (1, 128, 128):
            continue
        images.append(image)
        labels.append(3)

directory = os.fsencode("./Rock_Test")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = load_image("./Rock_Test/" + filename)[np.newaxis, :, :]
        # image = image[:, :, image.shape[2] // 2 - 64:image.shape[2] // 2 + 64]
        if image.shape != (1, 128, 128):
            continue
        images.append(image)
        labels.append(4)

data = np.concatenate(images, axis = 0)[...,np.newaxis]
labels = to_categorical(np.array(labels))

cnn_test_predictions = classifier.predict(data).tolist()

accuracy, confusion_matrix = evaluate(cnn_test_predictions, labels.tolist())
print("Testing accuracy:", accuracy)
print("Confusion matrix:", confusion_matrix)
