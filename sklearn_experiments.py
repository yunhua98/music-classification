import numpy as np
import matplotlib
from sklearn import linear_model, neural_network, datasets, neighbors, model_selection
from PIL import Image

def load_image(filename):
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def evaluate(predictions, labels):
    accuracy = sum(list(map(lambda x, y: 1 if x == y else 0, predictions, labels))) / len(labels)
    confusion_matrix = []
    for i in range(5):
        confusion_matrix.append([])
        for j in range(5):
            confusion_matrix[-1].append(sum(list(map(lambda y, x: 1 if x == i and y == j else 0, predictions, labels))))
    return accuracy, confusion_matrix

genres = ["Country", "EDM", "HipHop", "Latin", "Rock"]

images = []

num_sections = 11
for section in range(num_sections):
    for genre in genres:
        for i in range(10):
            filename = "./" + genre + "/" + str(i) + ".png"
            images.append(load_image(filename)[:, section * 128:(section + 1) * 128])

data = np.empty((len(images), images[0].size))

for instance, image in enumerate(images):
    index = 0
    for feature in np.nditer(image):
        data[instance, index] = feature
        index += 1

labels = np.empty((50*num_sections,))
for section in range(num_sections):
    for i in range(50):
        labels[section * 50 + i] = i // 10


validation_data = []

# for genre in genres:
#     for i in range(10):
#         filename = "./" + genre + "/" + str(i) + ".png"
#         validation_data.append(load_image(filename)[:, num_sections * 128 - 300:(num_sections + 1) * 128 - 300])
#         images.append(load_image(filename)[:, num_sections * 128 - 300:(num_sections + 1) * 128 - 300])

kf = model_selection.KFold(n_splits=11)

# print("Random Classifier")

# random_train_predictions = [np.random.randint(5) for _ in labels.tolist()]
# random_validation_predictions = [np.random.randint(5) for _ in range(50)]

# accuracy, confusion_matrix = evaluate(random_train_predictions, labels.tolist())
# print("Training accuracy:", accuracy)
# print("Confusion matrix:", confusion_matrix)

# accuracy, confusion_matrix = evaluate(random_validation_predictions, labels[:50].tolist())
# print("Validation accuracy:", accuracy)
# print("Confusion matrix:", confusion_matrix)


# print("\nLogistic Regression")

# logreg = linear_model.LogisticRegression(C=1e5)

# logreg.fit(data, labels)

# train_predictions = []

# for image in images:
#     train_predictions.append(logreg.predict(image.reshape((1, image.size))).flatten()[0])

# accuracy, confusion_matrix = evaluate(train_predictions, labels.tolist())
# print("Training accuracy:", accuracy)
# print("Confusion matrix:", confusion_matrix)

# validation_predictions = []

# for instance in validation_data:
#     validation_predictions.append(logreg.predict(instance.reshape((1, instance.size))).flatten()[0])

# accuracy, confusion_matrix = evaluate(validation_predictions, labels[:50].tolist())
# print("Validation accuracy:", accuracy)
# print("Confusion matrix:", confusion_matrix)


print("\nNeural Network")

neural_net = neural_network.MLPClassifier(hidden_layer_sizes=(128,128,128,), learning_rate_init=0.001, max_iter=1000)

training_accuracies = []
validation_accuracies = []

for train_index, validation_index in kf.split(data):
    neural_net.fit(data[train_index], labels[:500])

    nn_train_predictions = neural_net.predict(data[train_index]).tolist()

    # for image in images:
    #     nn_train_predictions.append(neural_net.predict(image.reshape((1, image.size))).flatten()[0])

    accuracy, confusion_matrix = evaluate(nn_train_predictions, labels[:500].tolist())
    training_accuracies.append(accuracy)
    print("Training accuracy:", accuracy)
    print("Confusion matrix:", confusion_matrix)

    nn_validation_predictions = neural_net.predict(data[validation_index])

    # for instance in validation_data:
    #     nn_validation_predictions.append(neural_net.predict(instance.reshape((1, instance.size))).flatten()[0])

    accuracy, confusion_matrix = evaluate(nn_validation_predictions, labels[:50].tolist())
    validation_accuracies.append(accuracy)
    print("Validation accuracy:", accuracy)
    print("Confusion matrix:", confusion_matrix)

print("Average training accuracy:", sum(training_accuracies) / len(training_accuracies))
print("Average validation accuracy:", sum(validation_accuracies) / len(validation_accuracies))



# print("\nKNN")

# knn = neighbors.KNeighborsClassifier(n_neighbors=3, p=1)

# training_accuracies = []
# validation_accuracies = []

# for train_index, validation_index in kf.split(data):
#     knn.fit(data[train_index], labels[:500])

#     knn_train_predictions = knn.predict(data[train_index]).tolist()

#     # for image in images:
#     #     knn_train_predictions.append(knn.predict(image.reshape((1, image.size))).flatten()[0])

#     accuracy, confusion_matrix = evaluate(knn_train_predictions, labels[:500].tolist())
#     training_accuracies.append(accuracy)
#     print("Training accuracy:", accuracy)
#     print("Confusion matrix:", confusion_matrix)

#     knn_validation_predictions = knn.predict(data[validation_index])

#     # for instance in validation_data:
#     #     knn_validation_predictions.append(neural_net.predict(instance.reshape((1, instance.size))).flatten()[0])

#     accuracy, confusion_matrix = evaluate(knn_validation_predictions, labels[:50].tolist())
#     validation_accuracies.append(accuracy)
#     print("Validation accuracy:", accuracy)
#     print("Confusion matrix:", confusion_matrix)

# print("Average training accuracy:", sum(training_accuracies) / len(training_accuracies))
# print("Average validation accuracy:", sum(validation_accuracies) / len(validation_accuracies))






