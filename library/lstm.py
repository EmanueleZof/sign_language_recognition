import os

from matplotlib import pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, LSTM, Dense

class LSTM_NN:
    """
    Questa classe rappresenta una rete neurale basata su LSTM per l'addestramento ed il riconoscimento di gesti (azioni) della ingua dei segni.
    
    Args:
        input_shape (tuple): La forma dell'input dei dati nella forma (lunghezza sequenza, dimensione caratteristiche).
        actions (int): Il numero di azioni diverse da prevedere.
        
    Attributes:
        input_shape (tuple): La forma dell'input dei dati.
        actions (int): Il numero di azioni diverse da prevedere.
        model (tf.keras.Model): Il modello della rete neurale.
        training_log (tf.keras.callbacks.History): Il registro dell'addestramento.
        
    Methods:
        compile(learning_rate=0.0001): Compila il modello per l'addestramento.
        train(X_train, y_train, X_val, y_val, epochs, batch_size): Addestra il modello.
        detect(X_test): Esegue la previsione su nuovi dati.
        summary(): Mostra un riassunto del modello.
        save(save_dir='./model'): Salva il modello su disco.
        load_weights(file_path): Carica i pesi del modello da un file.
        training_history(): Visualizza i grafici della storia dell'addestramento.
        _build_model(): Costruisce il modello della rete neurale.
        load(file_path='.'): Carica un modello completo o solo i pesi da un file.
    """
    def __init__(self,
                 input_shape,
                 actions):
        """
        Inizializza un oggetto della classe con la forma dell'input e il numero di azioni.

        Args:
            input_shape (tuple): La forma dell'input dei dati nella forma (lunghezza sequenza, dimensione caratteristiche).
            actions (int): Il numero di azioni diverse da prevedere.
        """
        self.input_shape = input_shape
        self.actions = actions

        self.model = None
        self.training_log = None

        self._build_model()

    def compile(self, learning_rate=0.0001):
        """
        Compila il modello specificando l'ottimizzatore, la funzione di costo e le metriche di valutazione.

        Args:
            learning_rate (float, optional): Il tasso di apprendimento dell'ottimizzatore Adam. Predefinito a 0.0001.
        """
        opt = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Addestra il modello utilizzando i dati di addestramento e restituisce un log dell'addestramento.

        Args:
            X_train (numpy.ndarray): Dati di addestramento di input.
            y_train (numpy.ndarray): Etichette dei dati di addestramento.
            X_val (numpy.ndarray): Dati di validazione di input.
            y_val (numpy.ndarray): Etichette dei dati di validazione.
            epochs (int): Numero di epoche di addestramento.
            batch_size (int): Dimensione del batch durante l'addestramento.

        Returns:
            TrainingHistory: Un log contenente informazioni sull'andamento dell'addestramento.
        """
        self.training_log = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs)

    def detect(self, X_test):
        """
        Esegue la previsione utilizzando il modello su un insieme di dati di test.

        Args:
            X_test (numpy.ndarray): Dati di test di input.

        Returns:
            numpy.ndarray: Matrice di previsioni del modello per i dati di test.
        """
        return self.model.predict(X_test)

    def summary(self):
        """
        Stampa un riassunto della struttura del modello.

        Questo metodo visualizza una rappresentazione testuale della struttura del modello, comprese le dimensioni dei livelli e il numero di parametri.

        Returns:
            None
        """
        self.model.summary()

    def save(self, save_dir='./model'):
        """
        Salva il modello su disco.

        Questo metodo salva il modello su disco in due formati: 'actions.keras' e 'actions.h5'.
        I file verranno memorizzati nella directory specificata o nella directory predefinita ('./model') e sovrascriveranno i file esistenti con lo stesso nome.

        Args:
            save_dir (str, optional): La directory in cui salvare il modello. Default è './model'.

        Returns:
            None
        """
        self.model.save(os.path.join(save_dir, 'actions.keras'), overwrite=True)
        self.model.save(os.path.join(save_dir, 'actions.h5'), overwrite=True)

    def load_weights(self, file_path):
        """
        Carica i pesi (precedentemente allenati) del modello da un file specificato.

        I pesi vengono utilizzati per inizializzare il modello con le configurazioni specificate.

        Args:
            file_path (str): Il percorso del file contenente i pesi preallenati.

        Returns:
            None
        """
        self.model.load_weights(file_path)

    @classmethod
    def load(cls, file_path='.'):
        """
        Carica un modello LSTM_NN da un file specificato.

        Il tipo di file (".keras" o ".h5") determinerà il metodo di caricamento appropriato.

        Args:
            file_path (str): Il percorso del file contenente il modello o i pesi preallenati.

        Returns:
            LSTM_NN: Un'istanza del modello LSTM_NN con pesi preallenati se il caricamento ha successo, altrimenti None.
        """
        if '.keras' in file_path:
            return load_model(file_path)
        elif '.h5' in file_path:
            model = LSTM_NN(input_shape=(100,1662), actions=3)
            model.load_weights(file_path)
            return model

    def training_history(self):
        """
        Visualizza la storia dell'addestramento del modello.

        Questo metodo genera due grafici: uno per la perdita (loss) e uno per l'accuratezza (accuracy) del modello
        durante l'addestramento. I dati di addestramento e di validazione vengono confrontati nei grafici.

        Returns:
            None
        """
        #print(self.training_log.history.keys())

        _, (loss, acc) = plt.subplots(1, 2, figsize=(20,6))

        # Loss
        loss.plot(self.training_log.history['loss'])
        loss.plot(self.training_log.history['val_loss'])
        loss.title.set_text('Model loss')
        loss.set_ylabel('loss')
        loss.set_xlabel('epoch')
        loss.legend(['train', 'validation'], loc='upper left')

        # Accuracy
        acc.plot(self.training_log.history['categorical_accuracy'])
        acc.plot(self.training_log.history['val_categorical_accuracy'])
        acc.title.set_text('Model accuracy')
        acc.set_ylabel('accuracy')
        acc.set_xlabel('epoch')
        acc.legend(['train', 'validation'], loc='upper left')

    def _build_model(self):
        """
        Costruisce il modello della rete.

        Questo metodo crea un modello di rete neurale profonda utilizzando i layer LSTM e Dense specificati con le 
        dimensioni appropriate. Il modello viene definito utilizzando l'API funzionale di Keras.

        Returns:
            None
        """
        input = Input(shape=self.input_shape)

        model = LSTM(64, return_sequences=True, activation='tanh')(input)
        model = LSTM(128, return_sequences=True, activation='tanh')(model)
        model = LSTM(64, return_sequences=False, activation='tanh')(model)
        model = Dense(64, activation='relu')(model)
        model = Dense(32, activation='relu')(model)
        model = Dense(self.actions, activation='softmax')(model)

        self.model = Model(input, model)
