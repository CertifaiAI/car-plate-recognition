'''
    Send status to kraftboard server -> Device status, Gate status, Entrance view
    Protocol => mqtt / http
'''
from utils.functions import cv2Img_base64Img
from threading import Thread
import paho.mqtt.client as mqtt
import json
import time

class StatusReport:
    def __init__(self, config, camera, door):
        self.stopped = False
        self.thread = None
        # Non-changable as long as device on
        self.device_status = 1
        # Changable if gate open
        self.gate_status = 0
        # Base 64 str of entrance view
        self.entrance_view = ''
        self.config = config
        # Initiate connection to mqtt
        self.client = mqtt.Client()
        self.client.username_pw_set(self.config.STATUS_TOKEN)

        try:
            self.client.connect(self.config.STATUS_HOST, self.config.STATUS_PORT)
            self.client.loop_start() 
        except:
            print('cannot connect to mqtt server')


        self.camera = camera
        self.doorControl = door

    def on_message(self, client, userdata, message):
        msg = message.payload.decode("utf-8")
        # string to dict
        msg = json.loads(msg)
        if msg['method']:
            if msg['method'] == 'deviceStatus':
                request_id = message.topic.split('/')[-1]
                self.client.publish(str(self.config.STATUS_END_POINT) + 'response/{}'.format(str(request_id)), json.dumps({ 'value': self.device_status}))
            elif msg['method'] == 'gateStatus':
                request_id = message.topic.split('/')[-1]
                self.client.publish(str(self.config.mqtt_endpoint_sub) + 'response/{}'.format(str(request_id)), json.dumps({ 'value': self.gate_status}))
            elif msg['method'] == 'view':
                request_id = message.topic.split('/')[-1]
                viewFrame = cv2Img_base64Img(self.camera.result())
                self.client.publish(str(self.config.mqtt_endpoint_sub) + 'response/{}'.format(str(request_id)), json.dumps({ 'value': viewFrame}))
            elif msg['method'] == 'openGate':
                # Open gate here
                self.doorControl.open()

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        print("Status Report Thread starting")
        return self

    def update(self):
        while True:
            if self.stopped:
                self.client.loop_stop()
                self.client.disconnect()
                return
            # Send request via mqtt / http every seconds
            self.client.loop_start()
            self.client.subscribe(str(self.config.mqtt_endpoint_sub)+'request/+')
            self.client.on_message = self.on_message
            time.sleep(1)


    def stop(self):       
        self.stopped = True

