#!pip install influxdb-client
#Import required libraries
import numpy as np
import time
from tqdm import tqdm
# from google.colab import drive
from sklearn.model_selection import train_test_split
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient

import matplotlib.pyplot as plt
import datetime
import calendar
import copy
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
# drive.mount('/content/drive')
# print('Libraries imported')


index_key = 0

def recuperation_donnees(time = '100y', salle = "d251"):
    #Define input and output
    path   = '/content/drive/My Drive/SAE capteur/'

    # Récupération des données
    url = "http://51.83.36.122:8086"
    token = "q4jqYhdgRHuhGwldILZ2Ek1WzGPhyctQ3UgvOII-bcjEkxqqrIIacgePte33CEjekqsymMqWlXnO0ndRhLx19g=="
    org = "INFO"
    bucket = "IUT_BUCKET"

    # Create a client
    client = InfluxDBClient(url=url, token=token, org=org, timeout=0)

    if salle == "d251":
        query = f'from(bucket: "IUT_BUCKET") |> range(start: -{time}) |> filter(fn: (r) => r["entity_id"] == "d251_1_multisensor_ultraviolet" or r["entity_id"] == "d251_1_multisensor_motion_detection" or r["entity_id"] == "d251_1_multisensor_air_temperature" or r["entity_id"] == "d251_1_co2_volatile_organic_compound_level" or r["entity_id"] == "d251_1_multisensor_humidity" or r["entity_id"] == "d251_1_multisensor_illuminance" or r["entity_id"] == "d251_1_co2_moderately_polluted" or r["entity_id"] == "d251_1_co2_slightly_polluted" or r["entity_id"] == "d251_1_co2_air_temperature" or r["entity_id"] == "d251_1_co2_dew_point" or r["entity_id"] == "d251_1_co2_carbon_dioxide_co2_level" or r["entity_id"] == "d251_1_co2_highly_polluted" or r["entity_id"] == "d251_1_co2_humidity") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["_measurement"] == "°C" or r["_measurement"] == "ppm" or r["_measurement"] == "lx" or r["_measurement"] == "%" or r["_measurement"] == "UV index") |> filter(fn: (r) => r["entity_id"] !~ /dew/) |> filter(fn: (r) => r["entity_id"] !~ /compound/) |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> yield(name: "mean")'

    elif salle == "d351":
        query = f'from(bucket: "IUT_BUCKET") |> range(start: -{time}) |> filter(fn: (r) => r["entity_id"] == "d351_3_co2_volatile_organic_compound_level" or r["entity_id"] == "d351_3_co2_humidity" or r["entity_id"] == "d351_3_co2_dew_point" or r["entity_id"] == "d351_3_co2_carbon_dioxide_co2_level" or r["entity_id"] == "d351_2_multisensor_ultraviolet" or r["entity_id"] == "d351_2_multisensor_motion_detection" or r["entity_id"] == "d351_3_co2_air_temperature" or r["entity_id"] == "d351_2_multisensor_illuminance" or r["entity_id"] == "d351_1_co2_air_temperature" or r["entity_id"] == "d351_1_co2_carbon_dioxide_co2_level" or r["entity_id"] == "d351_1_co2_dew_point" or r["entity_id"] == "d351_1_co2_highly_polluted" or r["entity_id"] == "d351_1_co2_humidity" or r["entity_id"] == "d351_1_co2_moderately_polluted" or r["entity_id"] == "d351_1_co2_slightly_polluted" or r["entity_id"] == "d351_1_co2_volatile_organic_compound_level" or r["entity_id"] == "d351_1_multisensor9_air_temperature" or r["entity_id"] == "d351_1_multisensor9_humidity" or r["entity_id"] == "d351_1_multisensor9_carbon_dioxide_co2_level" or r["entity_id"] == "d351_1_multisensor9_illuminance" or r["entity_id"] == "d351_1_multisensor9_loudness" or r["entity_id"] == "d351_1_multisensor9_particulate_matter_2_5" or r["entity_id"] == "d351_1_multisensor9_smoke_density" or r["entity_id"] == "d351_1_multisensor9_volatile_organic_compound_level" or r["entity_id"] == "d351_1_multisensor_air_temperature" or r["entity_id"] == "d351_1_multisensor_humidity" or r["entity_id"] == "d351_1_multisensor_illuminance" or r["entity_id"] == "d351_1_multisensor_motion_detection" or r["entity_id"] == "d351_1_multisensor_ultraviolet" or r["entity_id"] == "d351_2_co2_air_temperature" or r["entity_id"] == "d351_2_co2_highly_polluted" or r["entity_id"] == "d351_2_co2_carbon_dioxide_co2_level" or r["entity_id"] == "d351_2_co2_dew_point" or r["entity_id"] == "d351_2_co2_humidity" or r["entity_id"] == "d351_2_co2_slightly_polluted" or r["entity_id"] == "d351_2_co2_moderately_polluted" or r["entity_id"] == "d351_2_co2_volatile_organic_compound_level" or r["entity_id"] == "d351_2_multisensor_air_temperature" or r["entity_id"] == "d351_2_multisensor_humidity") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["_measurement"] == "µg/m³" or r["_measurement"] == "°C" or r["_measurement"] == "ppm" or r["_measurement"] == "lx" or r["_measurement"] == "dBA" or r["_measurement"] == "UV index" or r["_measurement"] == "%") |> filter(fn: (r) => r["entity_id"] !~ /dew/) |> filter(fn: (r) => r["entity_id"] !~ /compound/) |> filter(fn: (r) => r["entity_id"] !~ /smoke/) |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> yield(name: "mean")'

    elif salle == "d360":
        query = f'from(bucket: "IUT_BUCKET") |> range(start: -{time}) |> filter(fn: (r) => r["entity_id"] == "d360_1_multisensor_ultraviolet_2" or r["entity_id"] == "d360_1_multisensor_illuminance_2" or r["entity_id"] == "d360_1_multisensor_humidity_2" or r["entity_id"] == "d360_1_multisensor_motion_detection_2" or r["entity_id"] == "d360_1_multisensor_air_temperature_2" or r["entity_id"] == "d360_1_co2_moderately_polluted" or r["entity_id"] == "d360_1_co2_slightly_polluted" or r["entity_id"] == "d360_1_co2_volatile_organic_compound_level" or r["entity_id"] == "d360_1_co2_humidity" or r["entity_id"] == "d360_1_co2_highly_polluted" or r["entity_id"] == "d360_1_co2_carbon_dioxide_co2_level" or r["entity_id"] == "d360_1_co2_dew_point" or r["entity_id"] == "d360_1_co2_air_temperature") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["_measurement"] == "°C" or r["_measurement"] == "ppm" or r["_measurement"] == "lx" or r["_measurement"] == "UV index" or r["_measurement"] == "%") |> filter(fn: (r) => r["entity_id"] !~ /dew/) |> filter(fn: (r) => r["entity_id"] !~ /compound/)  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false) |> yield(name: "mean")'

    # Get the Query API
    query_api = client.query_api()

    # Execute the query
    result = query_api.query(org=org, query=query)

    # Close the client
    client.close()

    return result

def stockage_result(result):
    # Liste pour stocker tous les enregistrements
    all_records = []

    # Rassembler tous les enregistrements de toutes les tables
    for table in result:
        for record in table.records:
            all_records.append(record)

    # Trier la liste des enregistrements par le champ 'time'
    all_records.sort(key=lambda record: record.get_time())

    # Récupération des measurements
    unique_measurements = set()

    # Parcourir chaque enregistrement dans la table
    for record in all_records:
        # Ajouter la valeur de _measurement au set
        unique_measurements.add(record.get_measurement())

    # Afficher les valeurs uniques
    print(unique_measurements)

    # Rangement des données

    data_by_measurement = {}


    for measurement in unique_measurements:
        data_by_measurement[measurement] = {"time": [], "value": []}

    for record in all_records:
        data_by_measurement[record.get_measurement()]["value"].append(record.get_value())
        data_by_measurement[record.get_measurement()]["time"].append(record.get_time())


    print(len(data_by_measurement['lx']['value']))

    print(max(data_by_measurement['°C']['value']))

    return data_by_measurement


def traitement_donnees(data_by_measurement):
    # Nétoyage des données

    # Exemple de plages acceptables
    acceptable_ranges = {
        "%": (0, 100),
        "dBA": (0, 100),
        "lx": (0, 100),
        "µg/m³": (0, 100),
        "°C": (-50, 50),
        "ppm": (2, 7000),
        "UV index": (0, 11)
    }

    exclude_zero_for_measurements = {"%", "dBA", "°C", "ppm"}

    # Nettoyage des données
    for measurement, data in data_by_measurement.items():
        if measurement in acceptable_ranges:
            min_val, max_val = acceptable_ranges[measurement]

            # Filtrer les valeurs qui ne sont pas dans la plage acceptable
            data_by_measurement[measurement]["value"] = [val for val, time in zip(data["value"], data["time"]) if min_val <= val <= max_val and (val != 0 if measurement in exclude_zero_for_measurements else True)]
            data_by_measurement[measurement]["time"] = [time for val, time in zip(data["value"], data["time"]) if min_val <= val <= max_val and (val != 0 if measurement in exclude_zero_for_measurements else True)]

    # Afficher les données nettoyées
    print(len(data_by_measurement['lx']['value']))

    def calculate_normalized_time(dt):
        # Normaliser le numéro du mois (0 pour janvier, 1 pour décembre)
        normalized_month = (dt.month - 1) / 11

        # Normaliser le jour du mois
        days_in_month = calendar.monthrange(dt.year, dt.month)[1]  # Nombre de jours dans le mois
        normalized_day = (dt.day - 1) / (days_in_month - 1)

        # Normaliser l'heure du jour
        normalized_hour = (dt.hour * 60 + dt.minute) / (24 * 60)

        return normalized_month, normalized_day, normalized_hour


    # Normalisation des données

    data_by_measurement_normalized = copy.deepcopy(data_by_measurement)
    data_by_measurement_no_normalized = copy.deepcopy(data_by_measurement)

    for measurement, data in data_by_measurement_normalized.items():
        # Vérifiez que la liste des valeurs n'est pas vide
        if data['value']:
            max_value = max(data['value'])
            min_value = min(data['value'])
        else:
            # Dans le cas où il n'y a pas de données pour ce measurement
            max_value = None
            min_value = None

        data_by_measurement_no_normalized[measurement]["value"] = [val for val, time in zip(data["value"], data["time"])]
        data_by_measurement_no_normalized[measurement]["time"] = [calculate_normalized_time(time) for val, time in zip(data["value"], data["time"])]

        data_by_measurement_normalized[measurement]["value"] = [((val - min_value) / (max_value - min_value)) if max_value - min_value != 0 else 0 for val, time in zip(data["value"], data["time"])]
        data_by_measurement_normalized[measurement]["time"] = [calculate_normalized_time(time) for val, time in zip(data["value"], data["time"])]

    return data_by_measurement_normalized, data_by_measurement_no_normalized

def structure(data_by_measurement_normalized, data_by_measurement_no_normalized):
    def organisation(full_data):
        # Étape 1: Identifier toutes les unités de mesure
        units = list(full_data.keys())
        #print(len(full_data['ppm']['time'][0]))

        # Étape 2: Créer une liste de tous les temps uniques
        unique_times = set()
        for unit in units:
            unique_times.update(full_data[unit]['time'])
        unique_times = sorted(list(unique_times))

        # Étape 3: Construire le tableau final
        table = []

        # En-têtes pour le tableau
        headers = units + ['time_month', 'time_day', 'time_hour']
        table.append(headers)

        # Valeur par défaut si pas de données
        default_value = 0

        # Dictionnaire pour garder la trace des dernières valeurs connues pour chaque unité
        last_known_value = {unit: default_value for unit in units}

        for time_point in unique_times:
            row = []
            for unit in units:
                try:
                    index = full_data[unit]['time'].index(time_point)
                    value = full_data[unit]['value'][index]
                    last_known_value[unit] = value  # Mettre à jour la dernière valeur connue
                except ValueError:
                    value = last_known_value[unit]  # Utiliser la dernière valeur connue ou la valeur par défaut si aucune valeur précédente n'existe

                row.append(value)

            # Ajouter les informations temporelles
            row.extend(time_point)

            # Ajouter la ligne au tableau
            table.append(row)
        return table

    # 'table' contient maintenant les données transformées avec la logique de la dernière valeur connue.

    full_dataX = data_by_measurement_normalized
    full_dataY = data_by_measurement_no_normalized


    tableX = organisation(full_dataX)
    tableY = organisation(full_dataY)



    tableX = np.array(tableX)

    tableX = tableX.transpose()


    tableY = np.array(tableY)

    tableY = tableY.transpose()



    # Définition X,Y

    organized_dataX = tableX[:, 1:]
    organized_dataY = tableY[0:len(tableX)-3, 1:]

    return organized_dataX, organized_dataY

def data_split(organized_dataX, organized_dataY):

    n_input_steps = 5  # Nombre de pas de temps en entrée
    n_future_steps = 5  # Nombre de pas de temps que vous voulez prédire

    X, Y = [], []

    for i in range(1, len(organized_dataX[0]) - 1 - n_input_steps - n_future_steps + 1):
        X.append(organized_dataX[:, i:i + n_input_steps])
        Y.append(organized_dataY[:, i + n_input_steps:i + n_input_steps + n_future_steps])

    X = np.array(X)
    Y = np.array(Y)

    # Fractionnement des données


    # Définir les ratios pour la séparation
    test_ratio = 0.15
    validation_ratio = 0.15
    train_ratio = 1 - (test_ratio + validation_ratio)

    # Dictionnaires pour stocker les ensembles divisés
    train_set = {}
    validation_set = {}
    test_set = {}

    # Fractionnement initial en entraînement et test
    X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=(test_ratio + validation_ratio))

    # print(type(X), type(Y))

    # Fractionnement du temporaire en validation et test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio/(test_ratio + validation_ratio))

    # Stocker les ensembles divisés
    train_set = (X_train, y_train)
    validation_set = (X_val, y_val)
    test_set = (X_test, y_test)

    return X_train, X_temp, y_train, y_temp


# Pas besoin
def entrainement_model(model, X_train, y_train, X_temp, y_temp):

    #Entrainement du modèle

    X_trainf = X_train.astype('float32')

    y_trainf = y_train[:][:,index_key]
    y_trainf = y_trainf.astype('float32')



    # Définir un callback pour l'arrêt précoce pour éviter le surajustement
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, min_delta=0.0001, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001, verbose=1, min_delta=0.0001)

    # Entraîner le modèle
    history = model.fit(X_trainf, y_trainf, epochs=1000, validation_split=0.2, callbacks=[early_stopping]) #, batch_size=32

    # Visualiser l'historique de l'entraînement

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    X_test = X_temp.astype('float32')

    y_tempf = y_temp[:][:,index_key]
    y_test = y_tempf.astype('float32')


    # Évaluer le modèle sur les données de test
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("Test Loss: ", test_loss)
    print("Test MAE: ", test_mae)



    # Faire des prédictions
    predictions = model.predict(X_test)

    # S'assurer que les prédictions sont sous la bonne forme (1D)
    predictionsf = predictions.flatten()

    # Vérifier que y_test et predictions ont la même longueur
    assert len(y_test.flatten()) == len(predictionsf), "Les longueurs de y_test et predictions doivent être égales."

    # Tracer les prédictions par rapport aux valeurs réelles
    plt.figure(figsize=(12,6))  # Rendre le graphique plus grand

    plt.plot(y_test.flatten()[:100], label='Valeurs Réelles')  # Limiter le nombre de points pour une meilleure lisibilité
    plt.plot(predictionsf[:100], label='Prédictions', alpha=0.7)  # La transparence peut aider si les lignes se chevauchent

    plt.title('Comparaison des Valeurs Réelles et des Prédictions')  # Ajouter un titre
    plt.xlabel('Index des Échantillons')  # Étiqueter l'axe X
    plt.ylabel('Valeur')  # Étiqueter l'axe Y
    plt.legend()  # Ajouter une légende

    plt.show()  # Afficher le graphique

    chemin_sauvegarde  = './model/model.h5'
    model.save(chemin_sauvegarde)
    print(f"Modèle sauvegardé avec succès à {chemin_sauvegarde}")

    return model


def pred_split(organized_dataX):

    n_input_steps = 10  # Nombre de pas de temps en entrée
    X = []
    X.append(organized_dataX[:, -n_input_steps:])


    X = np.array(X)

    return X


def prediction(model, X):

    X_test = X.astype('float32')

    # Faire des prédictions
    predictions = model.predict(X_test)

    return predictions

    
def predict(key='°C', salle = "d251"):
    model = load_model(f'/app/ia/model/{salle}/model_{key}.h5')
    result = recuperation_donnees('10h', salle)
    data_by_measurement = stockage_result(result)

    # liste_des_cles = list(data_by_measurement.keys())
    # index_key = liste_des_cles.index(key)

    data_by_measurement_normalized, data_by_measurement_no_normalized = traitement_donnees(data_by_measurement)
    organized_dataX, organized_dataY = structure(data_by_measurement_normalized, data_by_measurement_no_normalized)
    X = pred_split(organized_dataX)
    tab_res = []

    pred = prediction(model,X) 
    for table in pred:
        for record in table:
                tab_res.append({
                    'value': record,

                })
    print("Les prédictions", tab_res)
    return tab_res