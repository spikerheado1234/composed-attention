#!/bin/bash

echo 'preparing virtual environment and installing dependencies'

python3 -m venv env && source env/bin/activate && pip3 install -r requirements.txt
