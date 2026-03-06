#!/bin/sh

python -m venv venv
. ./venv/bin/activate
pip install -r requiremets.txt
deactivate
