# Importazione delle librerie necessarie
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ============================
# 1. Caricamento dei dati
# ============================
host = "localhost"
port = "5432"
dbname = "DataScience"
user = "postgres"
password = "2430"

conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
query = "SELECT * FROM public.vew_mts_prov"
df = pd.read_sql_query(query, conn)
conn.close()

print("Dati caricati:")
print(df.head())

# Ordinamento del dataset per "postazione" e "giorno"
df = df.sort_values(by=["postazione", "giorno"]).reset_index(drop=True)

# Funzione per creare sequenze temporali con feature aggiuntive
def crea_sequenze(dati_postazione, lookback):
    """
    Crea sequenze temporali per il forecasting, includendo feature aggiuntive.
    :param dati_postazione: DataFrame contenente i dati di una singola postazione.
    :param lookback: Numero di giorni da usare come input per la previsione.
    :return: Sequenze (X) e target (y).
    """
    sequenze = []
    target = []
    
    for i in range(len(dati_postazione) - lookback):
        # Estrai le feature per la sequenza corrente
        X_seq_transiti = dati_postazione.iloc[i:i+lookback]["transiti"].values.flatten()
        X_seq_leggeri = dati_postazione.iloc[i:i+lookback]["trleggeri"].values.flatten()
        X_seq_pesanti = dati_postazione.iloc[i:i+lookback]["trpesanti"].values.flatten()
        X_seq_feriali = dati_postazione.iloc[i:i+lookback]["trferiali"].values.flatten()
        X_seq_festivi = dati_postazione.iloc[i:i+lookback]["trfestivi"].values.flatten()
        X_seq_ngiornosettimana = dati_postazione.iloc[i:i+lookback]["ngiornosettimana"].values.flatten()
        
        # Combina tutte le feature in un'unica sequenza
        X_seq = np.concatenate((X_seq_transiti, X_seq_leggeri, X_seq_pesanti, X_seq_feriali, X_seq_festivi, X_seq_ngiornosettimana))
        
        # Target (transiti totali del giorno successivo)
        y_target = dati_postazione.iloc[i+lookback]["transiti"]
        
        sequenze.append(X_seq)
        target.append(y_target)
    
    return np.array(sequenze), np.array(target)

# Parametri
lookback = 14  # Numero di giorni da usare per la previsione
forecast_date = pd.to_datetime("2019-12-31")  # Data per cui fare il forecast

# Lista per memorizzare i risultati
risultati = []

# Iterazione su ogni postazione
for postazione in df["postazione"].unique():
    # Filtra i dati per la postazione corrente
    dati_postazione = df[df["postazione"] == postazione].copy()
    
    # Se non ci sono abbastanza dati, salta questa postazione
    if len(dati_postazione) < lookback + 1:
        continue  
    
    # Creazione delle sequenze per il training
    X, y = crea_sequenze(dati_postazione, lookback)
    
    # Normalizzazione dei dati
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Divisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Definizione del modello MLP
    model = Sequential()
    model.add(Dense(128, input_dim=lookback * 6, activation="relu"))  # Input dimension aggiornata (6 feature)
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    
    # Compilazione del modello
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mean_squared_error")
    
    # Aggiunta di Early Stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitora la perdita sul validation set
        patience=10,         # Numero di epoche senza miglioramenti prima di fermare l'addestramento
        restore_best_weights=True  # Ripristina i pesi migliori alla fine
    )
    
    # Addestramento del modello con Early Stopping
    history = model.fit(
        X_train, y_train,
        epochs=100,          # Numero massimo di epoche
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],  # Aggiungi il callback EarlyStopping
        verbose=0
    )
    
    # Valutazione del modello sul test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Postazione: {postazione}, Test Loss: {test_loss}")
    
    # Previsioni sul test set
    y_pred = model.predict(X_test).flatten()
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Memorizzazione dei risultati
    risultati.append({
        "Postazione": postazione,
        "Test Loss": test_loss,
        "True Values": y_test_rescaled,
        "Predicted Values": y_pred_rescaled
    })
    
    # Forecasting per il 31 dicembre
    # Filtra i dati fino al giorno prima del forecast_date
    dati_postazione_forecast = dati_postazione[dati_postazione["giorno"] < forecast_date]
    
    # Crea le sequenze per il forecasting
    X_forecast, y_forecast = crea_sequenze(dati_postazione_forecast, lookback)
    X_forecast_scaled = scaler_X.transform(X_forecast)
    y_pred_forecast = model.predict(X_forecast_scaled).flatten()
    y_pred_forecast_rescaled = scaler_y.inverse_transform(y_pred_forecast.reshape(-1, 1)).flatten()
    
    # Confronto con i valori reali del 31 dicembre
    valore_reale_31_dicembre = dati_postazione[dati_postazione["giorno"] == forecast_date]["transiti"].values[0]
    
    print(f"Postazione: {postazione}, Predizione 31/12: {y_pred_forecast_rescaled[0]:.2f}, Valore Reale: {valore_reale_31_dicembre}")

# Plot dei risultati per ogni postazione
for risultato in risultati:
    plt.figure(figsize=(10, 5))
    plt.plot(risultato["True Values"], label="True Values", marker='o')
    plt.plot(risultato["Predicted Values"], label="Predicted Values", marker='x')
    plt.title(f"True vs Predicted Values - Postazione {risultato['Postazione']}")
    plt.xlabel("Samples")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.show()

