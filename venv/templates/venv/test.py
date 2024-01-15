from influxdb_client import InfluxDBClient
# Récupération des données
url = "http://51.83.36.122:8086"
token = "q4jqYhdgRHuhGwldILZ2Ek1WzGPhyctQ3UgvOII-bcjEkxqqrIIacgePte33CEjekqsymMqWlXnO0ndRhLx19g=="
org = "INFO"
bucket = "IUT_BUCKET"

# Create a client
client = InfluxDBClient(url=url, token=token, org=org)

# Create a query
# Create a query
query = 'from(bucket: "IUT_BUCKET")|> range(start: -15m)|> filter(fn: (r) => r["_measurement"] == "°C")|> filter(fn: (r) => r["_field"] == "value")|> filter(fn: (r) => r["domain"] == "sensor")|> filter(fn: (r) => r["entity_id"] =~ /d351_1_multisensor9_air_temperature|d351_1_multisensor_air_temperature|d351_2_multisensor_air_temperature|d351_3_co2_air_temperaturd360_1_multisensor_air_temperature_2/)|> filter(fn: (r) => r["entity_id"] != "d360_1_co2_dew_point")|> aggregateWindow(every: 1h, fn: mean, createEmpty: false)|> yield(name: "mean")'

# Get the Query API
query_api = client.query_api()

# Execute the query
result = query_api.query(org=org, query=query)

# Process the results
# for table in result:
#     for record in table.records:
#         print(f"Time: {record.get_time()}, Value: {record.get_value()}, Measurement: {record.get_measurement()}")

# Close the client
client.close()


# Liste pour stocker tous les enregistrements
all_records = []

# Rassembler tous les enregistrements de toutes les tables
for table in result:
    for record in table.records:
            all_records.append(record)

# Trier la liste des enregistrements par le champ 'time'
all_records.sort(key=lambda record: record.get_time())

# Maintenant, all_records contient tous les enregistrements triés par temps
for record in all_records:

    print(f"Time: {record.get_time()}, Value: {record.get_value()}, Measurement: {record.get_measurement()}")