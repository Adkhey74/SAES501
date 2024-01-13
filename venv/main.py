from flask import Flask,render_template,send_file,Response,json,jsonify
from influxdb_client import InfluxDBClient
from ia.prediction_temp import predict
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
    pred = predict(key="ppm")
    
    return render_template('home.html',active_page='Home')

@app.route('/graph.html')
def graph():



    # Par exemple, rendre un modèle avec render_template
    return render_template('graph.html', active_page='Graph')




@app.route('/get_data')
def get_data():
    # Create a client
    client = InfluxDBClient(url=url, token=token, org=org, timeout=20000)
    # Create a query
# Create a query
    # query = 'from(bucket: "IUT_BUCKET")|> range(start: -12h)|> filter(fn: (r) => r["_measurement"] == "°C")|> filter(fn: (r) => r["_field"] == "value")|> filter(fn: (r) => r["domain"] == "sensor")|> filter(fn: (r) => r["entity_id"] == "d351_1_multisensor9_air_temperature")|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)|> yield(name: "mean")'
    query = 'from(bucket: "IUT_BUCKET")|> range(start: -12h)|> filter(fn: (r) => r["_measurement"] == "ppm")|> filter(fn: (r) => r["_field"] == "value")|> filter(fn: (r) => r["domain"] == "sensor")|> filter(fn: (r) => r["entity_id"] == "d351_1_multisensor9_carbon_dioxide_co2_level")|> difference(nonNegative: true, columns: ["_value"], keepFirst: false)|> filter(fn: (r) => r._value >= 50)|> yield(name: "mean")'
    



    # Get the Query API
    query_api = client.query_api()

    # Execute the query
    result = query_api.query(org=org, query=query)

    # Process the results
    # for table in result:
    #     for record in table.records:
    #         print(f"Time: {record.get_time()}, Value: {record.get_value()}, Measurement: {record.get_measurement()}")

    # Close the client
    res = []

    # Rassembler tous les enregistrements de toutes les tables
    for table in result:
        for record in table.records:
            if record.get_measurement() == "°C":
                res.append({
                    'time': record.get_time(),
                    'value': record.get_value(),
                    'measurement': record.get_measurement(),
                    'entity':  record['entity_id'],
                    'nbvaleur': len(table.records)

                })
            elif record.get_measurement() == "ppm":
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
    for record in res:
        yield f"data: {json.dumps(record)}\n\n"
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

