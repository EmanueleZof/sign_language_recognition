import os
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, ConfusionMatrixDisplay


MAX_FRAME_NUM = 100 #videos be 100 frames in length


def create_folder(folder_path, verbose=False):
    """
    Crea una nuova cartella nella posizione specificata se non esiste già.

    Args:
        folder_path (str): Il percorso della cartella da creare.
        verbose (bool, optional): Se True, stampa un messaggio quando la cartella viene creata (default è False).

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if verbose:
            print('Folder created: ', folder_path)

def plot_confusion_matrices(prediction, labels, label_map):
    """
    Plotta le matrici di confusione (confusion matrix) per le previsioni rispetto alle etichette reali.

    Le previsioni e le etichette sono trasformate dal formato "one-hot" all'array con gli indici corrispondenti.

    Args:
        prediction (numpy.ndarray): Le previsioni del modello.
        labels (numpy.ndarray): Le etichette reali.
        label_map (dict): Un dizionario che mappa gli indici alle etichette.

    Returns:
        None
    """
    y_hat = np.argmax(prediction, axis=1).tolist()
    y_true = np.argmax(labels, axis=1).tolist()

    confusion_matrices = multilabel_confusion_matrix(y_true, y_hat)

    for index, confusion_matrix in enumerate(confusion_matrices):
        label_map_index = list(label_map)[index]
        disp = ConfusionMatrixDisplay(confusion_matrix)
        disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='vertical')
        disp.ax_.set_title(f'{label_map[label_map_index]}')
        plt.show()

def get_accuracy_score(prediction, labels):
    """
    Calcola il punteggio di accuratezza delle previsioni rispetto alle etichette reali.

    Le previsioni e le etichette sono trasformate dal formato "one-hot" all'array con gli indici corrispondenti.

    Args:
        prediction (numpy.ndarray): Le previsioni del modello.
        labels (numpy.ndarray): Le etichette reali.

    Returns:
        float: Il punteggio di accuratezza delle previsioni rispetto alle etichette reali.
    """
    y_hat = np.argmax(prediction, axis=1).tolist()
    y_true = np.argmax(labels, axis=1).tolist()
    return accuracy_score(y_true, y_hat)

