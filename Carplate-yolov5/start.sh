#!/bin/bash
systemctl restart nvargus-daemon
python3 main.py --show --nano --sensor
