import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape, num_classes):
    model = models.Sequential()

    # Definizione dell'oggetto Input come primo livello
    model.add(layers.Input(shape=input_shape))

    # Strato convoluzionale 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Strato convoluzionale 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Strato completamente connesso 1
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))

    # Strato completamente connesso 2 (output)
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Parametri della CNN
input_shape = (32, 32, 3)  # Dimensione delle immagini di input (altezza, larghezza, canali)
num_classes = 10  # Numero di classi nel dataset CIFAR-10

# Creazione della CNN
cnn_model = create_cnn(input_shape, num_classes)

# Salvataggio del modello
cnn_model.save('cnn_model.keras')  # Salva il modello in un file chiamato 'cnn_model.keras'
