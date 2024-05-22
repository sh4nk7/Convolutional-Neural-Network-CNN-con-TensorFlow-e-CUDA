import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Caricamento e preprocessamento del dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Stampa le dimensioni del dataset CIFAR-10
print("Dimensioni delle immagini di addestramento:", train_images.shape)
print("Dimensioni delle etichette di addestramento:", train_labels.shape)
print("Dimensioni delle immagini di test:", test_images.shape)
print("Dimensioni delle etichette di test:", test_labels.shape)

# Stampa le prime 5 etichette di addestramento
print("Prime 5 etichette di addestramento:", train_labels[:5])

# Stampa le prime 5 immagini di addestramento
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(train_images[i])
    plt.axis('off')
plt.show()

# Addestramento della CNN
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history = cnn_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Valutazione delle prestazioni
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
