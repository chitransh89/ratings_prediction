#!/bin/sh
gunicorn --chdir app main:app -w 4 --threads 8 -b 0.0.0.0:8000