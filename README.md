Implementazione di una Convolutional Neural Network (CNN) con TensorFlow e CUDA


Autore
Giuseppe Dimonte
MAT 367431
giuseppe.dimonte@studenti.unipr.it

Descrizione del Progetto
Questo progetto si concentra sull'implementazione di una Convolutional Neural Network (CNN) per la classificazione di immagini utilizzando TensorFlow e CUDA per il calcolo parallelo. L'obiettivo principale è migliorare le prestazioni di addestramento del modello sfruttando le potenzialità delle GPU.

Struttura del Progetto
Il progetto è organizzato nei seguenti file principali:

document.tex - Documento LaTeX con la descrizione dettagliata del progetto.
bibliografia.bib - File BibTeX contenente le fonti bibliografiche.
cifar10_cnn_training.py - Script Python per l'addestramento della CNN sul dataset CIFAR-10.
cifar10_cnn_training_parallel.py - Script Python per l'addestramento parallelo della CNN utilizzando CUDA.
model_evaluation_results.txt - File di testo contenente i risultati della valutazione del modello.
Installazione
Per eseguire questo progetto, assicurarsi di avere installato i seguenti requisiti:

Python 3.x
TensorFlow
CUDA Toolkit
cuDNN
Istruzioni per l'Esecuzione
1. Installazione di TensorFlow e CUDA
Assicurarsi di avere installato TensorFlow, CUDA e cuDNN. Seguire le istruzioni di installazione fornite da NVIDIA per configurare correttamente CUDA e cuDNN.

2. Caricamento e Preprocessamento del Dataset
Il dataset CIFAR-10 è utilizzato per l'addestramento e la valutazione del modello. Lo script cifar10_cnn_training.py include il codice per caricare e preprocessare il dataset.

3. Addestramento del Modello
Eseguire lo script cifar10_cnn_training.py per addestrare la CNN sul dataset CIFAR-10.

bash
Copia codice
python cifar10_cnn_training.py
4. Addestramento Parallelo con CUDA
Per sfruttare il calcolo parallelo su GPU, eseguire lo script cifar10_cnn_training_parallel.py.

bash
Copia codice
python cifar10_cnn_training_parallel.py
5. Valutazione delle Prestazioni
Dopo l'addestramento, le prestazioni del modello vengono valutate utilizzando il set di test. I risultati vengono salvati nel file model_evaluation_results.txt.

Architettura della CNN
La CNN implementata comprende i seguenti strati:

Strato Convoluzionale 1: 32 filtri, 3x3, ReLU, padding 'same', stride 1
Strato di Pooling 1: Max pooling, 2x2, stride 2
Strato Convoluzionale 2: 64 filtri, 3x3, ReLU, padding 'same', stride 1
Strato di Pooling 2: Max pooling, 2x2, stride 2
Strato Completamente Connesso 1: 128 unità, ReLU
Strato Completamente Connesso 2 (output): Numero di classi, Softmax
File Principali
cifar10_cnn_training.py
Script per l'addestramento del modello. Include il caricamento del dataset, la definizione dell'architettura della CNN, e l'addestramento del modello.

cifar10_cnn_training_parallel.py
Script per l'addestramento parallelo del modello utilizzando TensorFlow e CUDA. Include la strategia di distribuzione per sfruttare le GPU disponibili.

Valutazione delle Prestazioni
Le prestazioni del modello vengono valutate utilizzando metriche come accuratezza, precisione, richiamo e F1-score. I risultati vengono salvati nel file model_evaluation_results.txt.

Possibili Sviluppi
Confrontare le prestazioni con altri modelli di riferimento.
Analizzare gli errori commessi dal modello per identificare eventuali aree di miglioramento.
Implementare tecniche di regolarizzazione come dropout e regolarizzazione L1/L2 per migliorare le prestazioni del modello.
Monitorare le prestazioni del modello in produzione.
Conclusioni
Questo progetto dimostra l'efficacia del calcolo parallelo nell'addestramento di reti neurali profonde per la classificazione di immagini, sfruttando le potenzialità delle GPU attraverso CUDA.

Per ulteriori dettagli, consultare il file document.tex.
