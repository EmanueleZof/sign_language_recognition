import os
import cv2
import sys
import numpy as np
import mediapipe as mp

import sign_language_action_detection.library.utils as utils

from tqdm import tqdm
from matplotlib import pyplot as plt


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# Mediapipe helpers

def detect_landmarks(image, model):
    """
    Rileva i landmark del corpo in un frame video utilizzando il modello di Mediapipe.

    Durante l'elaborazione, l'immagine risulta in sola lettura, così da ottimizzare il consumo di memoria e risorse. 

    Args:
        image (numpy.ndarray): Il frame di video in formato BGR di OpenCV.
        model (mediapipe holistic model): Il modello Mediapipe per il rilevamento dei landmark.

    Returns:
        tuple: Una tupla contenente due elementi:
            - L'immagine con i landmark visualizzati.
            - I risultati del rilevamento dei landmark ottenuti dal modello Mediapipe.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
    
def _draw_landmarks(image, results):
    """
    Disegna i landmark del corpo sull'immagine del frame video utilizzando i risultati del rilevamento.

    Args:
        image (numpy.ndarray): L'immagine del frame di video su cui disegnare i landmark.
        results: I risultati del rilevamento dei landmark ottenuti dal modello Mediapipe.

    Returns:
        None
    """
    # Disegna le connessioni del viso
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

    # Disegna le connessioni della postura
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Disegna le connessioni della mano sinistra
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Disegna le connessioni della mano destra
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def _draw_styled_landmarks(image, results):
    """
    Disegna i landmark del corpo sull'immagine del frame di video con uno stile personalizzato utilizzando i risultati del rilevamento.

    Args:
        image (numpy.ndarray): L'immagine del frame di video su cui disegnare i landmark.
        results: I risultati del rilevamento dei landmark ottenuti dal modello Mediapipe.

    Returns:
        None
    """

    # Disegna le connessioni del viso con uno stile personalizzato
    mp_drawing.draw_landmarks(image, 
                              results.face_landmarks, 
                              mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                              )
    
    # Disegna le connessioni della postura con uno stile personalizzato
    mp_drawing.draw_landmarks(image, 
                              results.pose_landmarks, 
                              mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    
    # Disegna le connessioni della mano sinistra con uno stile personalizzato
    mp_drawing.draw_landmarks(image, 
                              results.left_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              )
    
    # Disegna le connessioni della mano destra con uno stile personalizzato
    mp_drawing.draw_landmarks(image, 
                              results.right_hand_landmarks, 
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )

def _get_pose_keypoints(landmarks):
    """
    Estrae i punti chiave della postura dai landmark e li appiattisce (flatten) per passarli al modello di rete neurale (LSTM).

    33 landmarks per 4 coordinate = 132

    Args:
        landmarks: I landmark della postura ottenuti dal rilevamento.

    Returns:
        numpy.ndarray: Un array numpy contenente i punti chiave della postura appiattiti.
    """
    if landmarks:
        pose = []
        for res in landmarks.landmark:
            pose.append(np.array([res.x, res.y, res.z, res.visibility]))
        return np.array(pose).flatten()
    
    return np.zeros(33*4)

def _get_face_keypoints(landmarks):
    """
    Estrae i punti chiave del volto dai landmark e li appiattisce (flatten) per passarli al modello di rete neurale (LSTM).

    486 landmarks per 3 coordinate = 1458

    Args:
        landmarks: I landmark del volto ottenuti dal rilevamento.

    Returns:
        numpy.ndarray: Un array numpy contenente i punti chiave del volto appiattiti.
    """
    if landmarks:
        face = []
        for res in landmarks.landmark:
            face.append(np.array([res.x, res.y, res.z]))
        return np.array(face).flatten()
    
    return np.zeros(468*3)

def _get_left_hand_keypoints(landmarks):
    """
    Estrae i punti chiave della mano sinistra dai landmark e li appiattisce (flatten) per passarli al modello di rete neurale (LSTM).

    21 landmarks per 3 coordinate = 63

    Args:
        landmarks: I landmark della mano sinistra ottenuti dal rilevamento.

    Returns:
        numpy.ndarray: Un array numpy contenente i punti chiave della mano sinistra appiattiti.
    """
    if landmarks:
        lh = []
        for res in landmarks.landmark:
            lh.append(np.array([res.x, res.y, res.z]))
        return np.array(lh).flatten()
    
    return np.zeros(21*3)

def _get_right_hand_keypoints(landmarks):
    """
    Estrae i punti chiave della mano destra dai landmark e li appiattisce (flatten) per passarli al modello di rete neurale (LSTM).

    21 landmarks per 3 coordinate = 63

    Args:
        landmarks: I landmark della mano destra ottenuti dal rilevamento.

    Returns:
        numpy.ndarray: Un array numpy contenente i punti chiave della mano destra appiattiti.
    """
    if landmarks:
        rh = []
        for res in landmarks.landmark:
            rh.append(np.array([res.x, res.y, res.z]))
        return np.array(rh).flatten()
    
    return np.zeros(21*3)
        
def extract_keypoints(results):
    """
    Estrae i punti chiave da tutti i landmark ottenuti dai risultati del rilevamento.

    1662 landmark per frame

    Args:
        results: I risultati del rilevamento dei landmark.

    Returns:
        numpy.ndarray: Un array numpy contenente tutti i punti chiave appiattiti da tutti i landmark.
    """
    pose = _get_pose_keypoints(results.pose_landmarks)
    face = _get_face_keypoints(results.face_landmarks)
    lh = _get_left_hand_keypoints(results.left_hand_landmarks)
    rh = _get_right_hand_keypoints(results.right_hand_landmarks)
    return np.concatenate([pose, face, lh, rh])


# Preprocessing helpers

def process_video(video, save_dir='.'):
    """
    Elabora un video acquisendo i landmark del corpo dai frame del video e salvandoli come file numpy.

    Se il video nn è della lunghezza prefissata, vengono aggiunti dei frame vuoti in fondo (in modo da uniformare alla stessa dimensione tutti i video del dataset).

    Args:
        video (str): Il percorso del video da elaborare.
        save_dir (str, optional): Il percorso della directory in cui salvare i file di landmark. Default è la directory corrente.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video)

    if (cap.isOpened() == False):
        sys.exit('Error opening video stream or file')
    
    #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for frame_num in range(utils.MAX_FRAME_NUM):
            ret, frame = cap.read()

            if ret == False:
                frame = np.zeros((512, 512, 3), np.uint8)
            
            image, results = detect_landmarks(frame, holistic)
            #_draw_landmarks(image, results)
            keypoints = extract_keypoints(results)

            utils.create_folder(save_dir)
            file_name = os.path.join(save_dir, str(frame_num))
            np.save(file_name, keypoints)

            #cv2.imshow(image)
            #cv2_imshow(image)
        cap.release()
        cv2.destroyAllWindows()

def process_all_files(files_dir, save_dir='.'):
    """
    Elabora tutti i file presenti in una directory specificata.

    Args:
        files_dir (str): Il percorso della directory contenente i file da elaborare.
        save_dir (str, optional): Il percorso della directory in cui salvare i risultati. Default è la directory corrente.

    Returns:
        None
    """
    for file in tqdm(os.listdir(files_dir)):
        file_name = file.replace('_color.mp4','').replace('_depth.mp4','')
        file_path = os.path.join(files_dir, file)
        save_path = os.path.join(save_dir, file_name)
        process_video(file_path, save_path)
