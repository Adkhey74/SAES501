from datetime import datetime, timezone, timedelta

# PPM
PPM_THRESHOLD_INCOMFORT = 1000
PPM_THRESHOLD_DANGER = 2000

# Decibel
DBA_THRESHOLD_INCOMFORT = 70
DBA_THRESHOLD_DANGER = 80

# Window Open
PPM_WINDOW_DIFFERENCE = 200
DEGREE_WINDOW_DIFFERENCE = 0.2

# Presence D351
DBA_PRESENCE_THRESHOLD = 50


def calculate_average(results):
    total = 0
    n = 0
    for table in results:
        for record in table.records:
            n +=1
            total += record.get_value()
    average = False
    if len(results) > 0:
        average = total / n
    return average


def build_query(bucket, room, measurement, start, exclude, end,last=False):
    date_start = get_date_n_minutes_later(start)

    if end == 0:
        query = f'from(bucket: "{bucket}") |> range(start: {date_start}) |> filter(fn: (r) => r["_measurement"] == "{measurement}") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["entity_id"] =~ /{room}/) '
    else:
        date_end = get_date_n_minutes_later(end)
        query = f'from(bucket: "{bucket}") |> range(start: {date_start}, stop: {date_end}) |> filter(fn: (r) => r["_measurement"] == "{measurement}") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["entity_id"] =~ /{room}/) '

    if exclude != "":
        query += f'|> filter(fn: (r) => r["entity_id"] !~ /{exclude}/)'
    if last:
        query += f'|> last()'
    return query


def get_date_n_minutes_later(minutes):
    now = datetime.utcnow()
    date_n_minutes_later = now - timedelta(minutes=minutes)
    formatted_date = date_n_minutes_later.strftime("%Y-%m-%dT%H:%M:%SZ")
    return formatted_date

# return -2 si Exception
# return -1 si aucun changement
# return 0 si comfort 
# return 1 si incomfort
# return 2 si danger
def ppm_is_discomfort(client, org, bucket, room):
    query = build_query(bucket=bucket, room=room, measurement="ppm", start=5, exclude="compound", end=0)

    discomfort = 0
    try:
        result = client.query_api().query(org=org, query=query)
        moy = calculate_average(result)
        # print(moy)
        if moy is not False:
            if moy >= PPM_THRESHOLD_DANGER:
                discomfort = 2
            elif moy >= PPM_THRESHOLD_INCOMFORT:
                discomfort = 1
            # print(discomfort)
        else:
            discomfort = -1

    except Exception as e:
        discomfort = -2
    return discomfort

# return -2 si Exception
# return -1 si aucun changement
# return 0 si comfort 
# return 1 si incomfort
# return 2 si danger
def dba_is_discomfort(client, org, bucket, room):
    query = build_query(bucket=bucket, room=room, measurement="dBA", start=5, exclude="", end=0)

    discomfort = 0
    try:
        result = client.query_api().query(org=org, query=query)
        moy = calculate_average(result)
        if moy is not False:
            if moy >= DBA_THRESHOLD_DANGER:
                discomfort = 2
            elif moy >= DBA_THRESHOLD_INCOMFORT:
                discomfort = 1
        else:
            discomfort = -1
    except Exception as e:
        discomfort = -2
    return discomfort

# return -2 si Exception
# return -1 si aucun changement
# return 0 si fenêtre ouverte 
# return 1 si fenêtre fermée
def window_close(client, org, bucket, room, last_data=0):
    query_5_minutes_degree = build_query(bucket=bucket, room=room, measurement="°C", start=5, exclude="dew", end=0,last=True)
    query_10_minutes_degree = build_query(bucket=bucket, room=room, measurement="°C", start=10, exclude="dew", end=5)
    # query_last_degree = build_query(bucket=bucket, room=room, measurement="°C", start=0, exclude="dew", end=5)

    query_5_minutes_ppm = build_query(bucket=bucket, room=room, measurement="ppm", start=5, exclude="compound", end=0, last=True)
    query_10_minutes_ppm = build_query(bucket=bucket, room=room, measurement="ppm", start=10, exclude="compound", end=5)
    # query_last_ppm = build_query(bucket=bucket, room=room, measurement="ppm", start=0, exclude="compound", end=5)

    window_status = 2
    try:
        result_5_minutes_degree = client.query_api().query(org=org, query=query_5_minutes_degree)
        result_10_minutes_degree = client.query_api().query(org=org, query=query_10_minutes_degree)
        result_5_minutes_ppm = client.query_api().query(org=org, query=query_5_minutes_ppm)
        result_10_minutes_ppm = client.query_api().query(org=org, query=query_10_minutes_ppm)

        moy_5_minutes_degree = calculate_average(result_5_minutes_degree)
        moy_10_minutes_degree = calculate_average(result_10_minutes_degree)
        moy_5_minutes_ppm = calculate_average(result_5_minutes_ppm)
        moy_10_minutes_ppm = calculate_average(result_10_minutes_ppm)

        if moy_5_minutes_degree is False or moy_5_minutes_degree == moy_5_minutes_ppm:
            window_status = -1
            return window_status

        if moy_10_minutes_degree is False:
            result_last_degree = client.query_api().query(org=org, query=query_last_degree)
            moy_10_minutes_degree = calculate_average(result_last_degree)

        if moy_10_minutes_ppm is False:
            result_last_ppm = client.query_api().query(org=org, query=query_last_ppm)
            moy_10_minutes_ppm = calculate_average(result_last_ppm)

        drop_ppm = False
        if (moy_10_minutes_ppm - PPM_WINDOW_DIFFERENCE) >= moy_5_minutes_ppm:
            drop_ppm = True

        drop_degree = False
        if (moy_10_minutes_degree - DEGREE_WINDOW_DIFFERENCE) >= moy_5_minutes_degree:
            drop_degree = True

        if drop_degree and drop_ppm:
            window_status = 0

        if last_data == 1:
            drop_ppm = False
            if (moy_10_minutes_ppm + PPM_WINDOW_DIFFERENCE) >= moy_5_minutes_ppm:
                drop_ppm = True

            drop_degree = False
            if (moy_10_minutes_degree + DEGREE_WINDOW_DIFFERENCE) >= moy_5_minutes_degree:
                drop_degree = True

            if drop_degree and drop_ppm:
                window_status = 0

    except Exception as e:
        window_status = -2
    return window_status

# return -2 si Exception
# return -1 si aucun changement
# return 0 si mouvement 
# return 1 si aucun mouvement
def movement_here(client, org, bucket, room):
    date_minus_5_minutes = get_date_n_minutes_later(5)

    query = f'from(bucket: "{bucket}") |> range(start: {date_minus_5_minutes}) |> filter(fn: (r) => r["_measurement"] =~ /{room}/) |> filter(fn: (r) => r["_measurement"] =~ /motion/) |> filter(fn: (r) => r["_field"] == "value")  |> filter(fn: (r) => r["domain"] == "binary_sensor")'

    here = 2
    try:
        result = client.query_api().query(org=org, query=query)
        moy = calculate_average(result)
        if moy is not False:
            if moy >= 0:
                here = 0
        else:
            here = -1
    except Exception as e:
        here = -2
    return here

# return -2 si Exception
# return 0 si quelqu'un
# return 1 si personne
def presence_d351(client, org, bucket, last_data=0):
    room = "d351"
    date_7_days = get_date_n_minutes_later(10080)
    date_5_minutes = get_date_n_minutes_later(5)
    query_7_days_ppm = f'from(bucket: "{bucket}") |> range(start: {date_7_days}, stop:{date_5_minutes}) |> filter(fn: (r) => r["_measurement"] == "ppm") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["entity_id"] !~ /compound/) |> filter(fn: (r) => r["entity_id"] =~ /{room}/)'
    query_5_minutes_ppm = f'from(bucket: "{bucket}") |> range(start: {date_5_minutes}) |> filter(fn: (r) => r["_measurement"] == "ppm") |> filter(fn: (r) => r["_field"] == "value")  |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["entity_id"] !~ /compound/) |> filter(fn: (r) => r["entity_id"] =~ /{room}/)'

    query_7_days_degree = f'from(bucket: "{bucket}") |> range(start: {date_7_days}, stop:{date_5_minutes}) |> filter(fn: (r) => r["_measurement"] == "°C") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["entity_id"] !~ /dew/) |> filter(fn: (r) => r["entity_id"] =~ /{room}/)'
    query_5_minutes_degree = f'from(bucket: "{bucket}") |> range(start: {date_5_minutes}) |> filter(fn: (r) => r["_measurement"] == "°C") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["entity_id"] !~ /dew/) |> filter(fn: (r) => r["entity_id"] =~ /{room}/)'

    query_5_minutes_dba = f'from(bucket: "{bucket}") |> range(start: {date_5_minutes}) |> filter(fn: (r) => r["_measurement"] == "dBA") |> filter(fn: (r) => r["_field"] == "value") |> filter(fn: (r) => r["domain"] == "sensor") |> filter(fn: (r) => r["entity_id"] =~ /{room}/)'

    detect_status = 2
    try:
        result_7_days_ppm = client.query_api().query(org=org, query=query_7_days_ppm)
        result_5_minutes_ppm = client.query_api().query(org=org, query=query_5_minutes_ppm)
        result_7_days_degree = client.query_api().query(org=org, query=query_7_days_degree)
        result_5_minutes_degree = client.query_api().query(org=org, query=query_5_minutes_degree)
        result_5_minutes_dba = client.query_api().query(org=org, query=query_5_minutes_dba)

        moy_7_days_ppm = calculate_average(result_7_days_ppm)
        moy_5_minutes_ppm = calculate_average(result_5_minutes_ppm)
        moy_7_days_degree = calculate_average(result_7_days_degree)
        moy_5_minutes_degree = calculate_average(result_5_minutes_degree)
        moy_5_minutes_dba = calculate_average(result_5_minutes_dba)

        count_above_threshold = 0

        if moy_5_minutes_ppm is not False and moy_5_minutes_ppm > moy_7_days_ppm:
            count_above_threshold += 1

        if moy_5_minutes_degree is not False and moy_5_minutes_degree > moy_7_days_degree:
            count_above_threshold += 1

        if moy_5_minutes_dba is not False and moy_5_minutes_dba > DBA_PRESENCE_THRESHOLD:
            count_above_threshold += 1

        if count_above_threshold >= 2:
            detect_status = 0

    except Exception as e:
        detect_status = -2
    return detect_status
