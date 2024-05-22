import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Caricamento e preprocessamento del dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Definizione della strategia per l'addestramento su GPU
strategy = tf.distribute.MirroredStrategy()

# Creazione del modello all'interno del contesto della strategia
with strategy.scope():
    model = create_cnn(input_shape, num_classes)

    # Compilazione del modello
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Addestramento del modello all'interno del contesto della strategia
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
