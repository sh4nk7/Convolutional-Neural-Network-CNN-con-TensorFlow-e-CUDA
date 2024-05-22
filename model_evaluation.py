import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Utilizzo del modello per fare previsioni sul set di test
predictions = model.predict(test_images)

# Calcolo dell'accuratezza del modello
accuracy = accuracy_score(test_labels, np.argmax(predictions, axis=1))

# Calcolo della precisione, del richiamo e della F1-score
precision = precision_score(test_labels, np.argmax(predictions, axis=1), average='weighted')
recall = recall_score(test_labels, np.argmax(predictions, axis=1), average='weighted')
f1 = f1_score(test_labels, np.argmax(predictions, axis=1), average='weighted')

# Esame dei pattern e degli errori comuni
conf_matrix = confusion_matrix(test_labels, np.argmax(predictions, axis=1))

# Scrivi i risultati su un file esterno
with open('model_evaluation_results.txt', 'w') as file:
    file.write("Accuratezza del modello: {}\n".format(accuracy))
    file.write("Precisione del modello: {}\n".format(precision))
    file.write("Richiamo del modello: {}\n".format(recall))
    file.write("F1-score del modello: {}\n\n".format(f1))
    file.write("Matrice di confusione:\n")
    np.savetxt(file, conf_matrix, fmt='%d')
