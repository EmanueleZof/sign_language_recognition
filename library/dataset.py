import os
import shutil
import numpy as np
import pandas as pd

import sign_language_action_detection.library.utils as utils

from tensorflow.keras.utils import to_categorical


def get_label_map(file_path):
    """
    Ottiene una mappa delle etichette da un file CSV.

    La colonna con le etichette in lingua turca viene omessa.

    Vedi la funzione: cherry_pick_labels()

    Args:
        file_path (str): Il percorso del file CSV contenente le etichette.

    Returns:
        dict: Un dizionario che mappa gli ID delle classi alle etichette.
    """
    df = _get_dataframe(file_path)
    df.drop('TR', inplace=True, axis=1)
    return dict(df.values)

def get_features(folder_path, labels_path, labels_map):
    """
    Estrae le caratteristiche e le etichette dai dati delle sequenze e le converte in formati utili.

    Args:
        folder_path (str): Il percorso della cartella contenente le sequenze di dati.
        labels_path (str): Il percorso del file contenente le etichette delle sequenze.
        labels_map (dict): Il dizionario che mappa gli ID delle classi alle etichette.

    Returns:
        tuple: Una tupla contenente tre elementi:
            - Un array numpy bidimensionale con le caratteristiche estratte dalle sequenze.
            - Un array numpy bidimensionale con le etichette codificate in formato one-hot.
            - Un array numpy unidimensionale con le etichette originali.
    """
    features, labels = [], []

    for sequence in os.listdir(folder_path):
        window = []
        for frame in range(utils.MAX_FRAME_NUM):
            file_path = utils.os.path.join(folder_path, sequence, '{}.npy'.format(frame))
            res = np.load(file_path)
            window.append(res)
        features.append(window)
        df = _get_labels_dataframe(labels_path)
        class_id = df.loc[df['SampleName'] == sequence]['ClassID'].values[0]
        label_index = list(labels_map).index(class_id)
        labels.append(label_index)

    features = np.array(features)
    labels = np.array(labels)

    labels_one_hot_encoded = to_categorical(labels).astype(int)

    return features, labels_one_hot_encoded, labels

def get_label(label_map, index):
    """
    Ottiene l'etichetta corrispondente a un determinato indice dal dizionario delle etichette.

    Args:
        label_map (dict): Un dizionario che mappa gli indici alle etichette.
        index (int): L'indice per cui ottenere l'etichetta corrispondente.

    Returns:
        str: L'etichetta corrispondente all'indice specificato.
    """
    return label_map[list(label_map)[index]]

def _get_dataframe(file_path):
    """
    Legge un file CSV e restituisce un DataFrame.

    Args:
        file_path (str): Il percorso del file CSV da leggere.

    Returns:
        pandas.DataFrame: Un DataFrame contenente i dati dal file CSV.
    """
    df = pd.read_csv(file_path)
    return df

def _get_labels_dataframe(file_path):
    """
    Legge un file CSV contenente informazioni sulle etichette e restituisce un DataFrame.

    Args:
        file_path (str): Il percorso del file CSV contenente le informazioni sulle etichette.
        
    Returns:
        pandas.DataFrame: Un DataFrame contenente due colonne: 'SampleName' e 'ClassID'. Le etichette sono ordinate per 'ClassID'.
    """
    df = pd.read_csv(file_path, header=None)
    df = df.rename(columns={0: 'SampleName', 1: 'ClassID'})
    df = df.sort_values('ClassID')
    return df


# Cherry picking

def _get_classes_samples(df, labels_id):
    """
    Estrae i nomi delle sequenze appartenenti alle classi specificate da un DataFrame.

    Args:
        df (pandas.DataFrame): Il DataFrame contenente informazioni sulle sequenze.
        labels_id (list): Una lista di ID di classi da cui estrarre i nomi delle sequenze.

    Returns:
        numpy.ndarray: Un array numpy contenente i nomi delle sequenze appartenenti alle classi specificate.
    """
    subset = df.loc[df['ClassID'].isin(labels_id)]
    return subset['SampleName'].to_numpy()

def _scaffold_folders(folder_path):
    """
    Crea una struttura di cartelle di base in una directory principale.

    Args:
        folder_path (str): Il percorso della cartella principale da creare.

    Returns:
        str: Il percorso della sottocartella 'color' appena creata.
    """
    sub_folder = os.path.join(folder_path, 'color')
    utils.create_folder(folder_path, verbose=True)
    utils.create_folder(sub_folder, verbose=True)
    return sub_folder

def _move_files(files, source, destination):
    """
    Sposta una lista di file dalla cartella di origine a quella di destinazione.

    Args:
        files (list): Una lista di nomi di file da spostare.
        source (str): Il percorso della cartella di origine.
        destination (str): Il percorso della cartella di destinazione.

    Returns:
        None
    """
    for f in files:
        file_name = f+'_color.mp4'
        source_path = os.path.join(source, file_name)
        destination_path = os.path.join(destination, file_name)
        shutil.move(source_path, destination_path)

def _get_subset(labels, samples_id, source_path, destination_path):
    """
    Estrae un sottoinsieme di campioni dalla cartella di origine e li sposta in una cartella di destinazione.

    Args:
        labels (str): Il percorso del file CSV contenente le etichette.
        samples_id (list): Una lista di ID di campioni da estrarre.
        source_path (str): Il percorso della cartella di origine dei campioni.
        destination_path (str): Il percorso della cartella di destinazione per il sottoinsieme.

    Returns:
        tuple: Una tupla contenente due elementi:
            - Una lista dei nomi dei campioni estratti.
            - Il percorso della cartella di destinazione in cui sono stati spostati i campioni.
    """
    subset = []
    subset_folder = ''

    if os.path.exists(labels):
        df = _get_labels_dataframe(labels)
        subset = _get_classes_samples(df, samples_id)
        subset_folder = _scaffold_folders(destination_path)
        _move_files(subset, source_path, subset_folder)

    return subset, subset_folder

def cherry_pick_classes(classes, dataset):
    """
    Estrae e organizza un sottoinsieme di classi da un dataset in una nuova struttura di cartelle.

    Args:
        classes (list): Una lista di nomi di classi da estrarre dal dataset.
        dataset (dict): Un dizionario contenente i percorsi dei file e le informazioni del dataset.

    Returns:
        tuple: Una tupla contenente i percorsi delle cartelle contenenti i sottoinsiemi di allenamento, test e validazione.
    """
    utils.create_folder('./Dataset', verbose=True)
    
    # Classes
    classes_df = _get_dataframe(dataset['classes'])
    classes_subset = classes_df.loc[classes_df['EN'].isin(np.array(classes))]
    labels_id = classes_subset['ClassId'].values

    # Train
    train_subset, train_subset_folder = _get_subset(dataset['train']['labels'], labels_id, dataset['train']['color'], './Dataset/train')

    # Test
    test_subset, test_subset_folder = _get_subset(dataset['test']['labels'], labels_id, dataset['test']['color'], './Dataset/test')

    # Validation
    val_subset, val_subset_folder = _get_subset(dataset['val']['labels'], labels_id, dataset['val']['color'], './Dataset/val')
        
    return train_subset_folder, test_subset_folder, val_subset_folder

def cherry_pick_labels(file_path, classes):
    """
    Estrae un sottoinsieme di etichette da un file CSV in base alle classi specificate.

    La colonna con le etichette in lingua turca viene omessa.

    Vedi la funzione: get_label_map()

    Args:
        file_path (str): Il percorso del file CSV contenente le etichette.
        classes (list): Una lista di nomi di classi da estrarre dal file.

    Returns:
        dict: Un dizionario che mappa gli ID delle classi alle etichette del sottoinsieme.
    """
    df = _get_dataframe(file_path)
    df.drop('TR', inplace=True, axis=1)
    sub = df.loc[df['EN'].isin(classes)]
    return dict(sub.values)