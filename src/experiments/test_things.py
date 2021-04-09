import requests
import configparser
import datetime
import json
# Config Parser
config = configparser.ConfigParser()
config.read('../settings.ini')
# get backend config
backend_hostname = config['Backend']['hostname']
backend_port = config['Backend']['port']
backend_endpoint_getplates = config['Backend']['get_plate_endpoint']
backend_endpoint_car = config['Backend']['car_endpoint']
backend_endpoint_plate = config['Backend']['plate_endpoint']
#backend_endpoint_ai = config['Backend']['ai']
get_plate_address = 'http://{}:{}/{}'.format(backend_hostname, backend_port, backend_endpoint_getplates)
address = 'http://{}:{}'.format(backend_hostname, backend_port)
# thingsboard 
thingsboard_hostname = config['ThingsBoard']['hostname']
thingsboard_port = config['ThingsBoard']['port']
thingsboard_endpoint = config['ThingsBoard']['endpoint']
thingsboard_entrytoken = config['ThingsBoard']['entry_token']
thingsboard_exittoken = config['ThingsBoard']['exit_token']
thingsboard_extra = config['ThingsBoard']['extra_end']

entryParking = 'http://' + 'localhost'+ ':' + thingsboard_port + thingsboard_endpoint + thingsboard_entrytoken + thingsboard_extra
exitParking = 'http://' + thingsboard_hostname +':' + thingsboard_port + thingsboard_endpoint + thingsboard_exittoken + thingsboard_extra

current_time = str(datetime.datetime.now())
# print(entry)
# records = {'plate number': str('RTX3060'), 'enter_time': current_time}
records = {'plate number': str('RTX3060'), 'exit_time': current_time}
# response = requests.post(entry, data=json.dumps(entry_records))
try:    
    response = requests.post(exitParking, data=json.dumps(records))
    # print(response)
except:
    print('Cannot connect to thingsboard server')

    # curl -v -X POST -d "{\"temperature\": 25}" https://demo.thingsboard.io/api/v1/$ACCESS_TOKEN/telemetry --header "Content-Type:application/json" 
