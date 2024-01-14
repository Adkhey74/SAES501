from flask import Flask,render_template,send_file,Response,json,jsonify
from influxdb_client import InfluxDBClient
from ia.prediction_temp import predict
from inconfort.inconfort import ppm_is_discomfort,dba_is_discomfort,window_open,movement_here,presence_d351
# Récupération des données
url = "http://51.83.36.122:8086"
token = "q4jqYhdgRHuhGwldILZ2Ek1WzGPhyctQ3UgvOII-bcjEkxqqrIIacgePte33CEjekqsymMqWlXnO0ndRhLx19g=="
org = "INFO"
bucket = "IUT_BUCKET"

app = Flask(__name__)



# Permet de régler les problèmes de print
import builtins

original_print = builtins.print

def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)

builtins.print = print


@app.route('/')
def home():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template

    # Possible keys = ["%","dBA", "lx", "µg/m³", "°C", "ppm", "UV index"] / UV index pas précis car les données sont très mauvaises
    # pred = predict(key="ppm")
    
    return render_template('home.html',active_page='Home')

@app.route('/graph.html')
def graph():



    # Par exemple, rendre un modèle avec render_template
    return render_template('graph.html', active_page='Graph')

last_data_by_room = {"d251":0,"d351":0,"d360":0}
client = InfluxDBClient(url=url, token=token, org=org, timeout=20000)
@app.route('/get_data/<room>')
def get_comfort(room):
    global client
    global last_data_by_room
    room = room.lower()
    last_data = last_data_by_room[room]
    comfortPpm = ppm_is_discomfort(client,org,bucket,room)
    comfortDba = dba_is_discomfort(client,org,bucket,room)
    comfortHere = movement_here(client,org,bucket,room)
    comfortwindow = window_open(client,org,bucket,room,last_data)
    
    presenceD351 = 0
    if room == "d351":
        presenceD351 = presence_d351(client,org,bucket)

    if comfortwindow == 1:
        last_data_by_room[room] = 1

    return jsonify(comfortPpm=comfortPpm,comfortDba=comfortDba,
                   comfortHere=comfortHere,comfortwindow=comfortwindow,presenceD351=presenceD351)


@app.route('/get_data/<salle>/<unite>/<temps>')
def get_data(salle, unite,temps):
    # Create a client
    client = InfluxDBClient(url=url, token=token, org=org, timeout=20000)
    # Create a query
# Create a query
    if unite.lower() == "co2":
        entity_id = f"{salle}_1_co2_carbon_dioxide_co2_level"
        measurement = 'ppm'
    elif unite.lower() == "temperature":
        entity_id = f"{salle}_1_co2_air_temperature"
        measurement = '°C'

    elif unite.lower() == "luminosite":
        if salle == 'd360':
            entity_id = f"{salle}_1_multisensor_illuminance_2"
        else:
            entity_id = f"{salle}_1_multisensor_illuminance"
        measurement = 'lx'
    elif unite.lower() == "bruit":
        entity_id = f"{salle}_1_multisensor9_loudness"
        measurement = 'dBA'
    elif unite.lower() == "humidite":
        entity_id = f"{salle}_1_co2_humidity"
        measurement = '%'
    elif unite.lower() == "uv":
        entity_id = f"{salle}_1_multisensor_ultraviolet"
        measurement = 'UV index'
    elif unite.lower() == "µg":
        entity_id = f"{salle}_1_multisensor9_particulate_matter_2_5"
        measurement = 'µg/m³'

    # query = 'from(bucket: "IUT_BUCKET")|> range(start: -12h)|> filter(fn: (r) => r["_measurement"] == "°C")|> filter(fn: (r) => r["_field"] == "value")|> filter(fn: (r) => r["domain"] == "sensor")|> filter(fn: (r) => r["entity_id"] == "d351_1_multisensor9_air_temperature")|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)|> yield(name: "mean")'
    query = f'from(bucket: "IUT_BUCKET")|> range(start: -{temps} )|> filter(fn: (r) => r["_measurement"] == "{measurement}")|> filter(fn: (r) => r["_field"] == "value")|> filter(fn: (r) => r["domain"] == "sensor")|> filter(fn: (r) => r["entity_id"] == "{entity_id}")|> yield(name: "mean")'
    print(query)


    # Get the Query API
    query_api = client.query_api()

    # Execute the query
    result = query_api.query(org=org, query=query   )

    # Process the results
    # for table in result:
    #     for record in table.records:
    #         print(f"Time: {record.get_time()}, Value: {record.get_value()}, Measurement: {record.get_measurement()}")

    # Close the client
    res = []

    # Rassembler tous les enregistrements de toutes les tables
    for table in result:
        for record in table.records:
                res.append({
                    'time': record.get_time(),
                    'value': record.get_value(),
                    'measurement': record.get_measurement(),
                    'entity':  record['entity_id'],
                    'nbvaleur': len(table.records)

                })



    res.sort(key=lambda record: record['time'])
    
    return Response(generate(res),content_type='text/event-stream')

def generate(res):
    print(res)
    yield f"data: {json.dumps(res)}\n\n"

@app.route('/tableau.html')
def tableau():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('tableau.html',active_page='Tableau')

@app.route('/detection_inconfort.html')
def detection():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('detection_inconfort.html',active_page='Detection')

@app.route('/connexion_compte.html')
def connexion():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('connexion_compte.html',active_page='Connexion')
@app.route('/configuration.html')
def configuration():
    # Votre logique de traitement peut être ajoutée ici
    # Par exemple, rendre un modèle avec render_template
    return render_template('configuration.html',active_page='Configuration')








if __name__ == '__main__':
    app.run(debug=True, port=34, host='0.0.0.0')

