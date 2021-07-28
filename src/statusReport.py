'''
    Send status to kraftboard server -> Device status, Gate status, Entrance view
    Protocol => mqtt / http
'''

from threading import Thread
class StatusReport:
    def __init__(self, config):
        self.stopped = False
        self.thread = None
        # Non-changable as long as device on
        self.device_status = 'on'
        # Changable if gate open
        self.gate_status = 'off'
        # Base 64 str of entrance view
        self.entrance_view = ''
        self.config = config

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        print("Status Report Thread starting")
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            # Send request via mqtt / http every seconds
            data = {}


    def stop(self):       
        self.stopped = True

