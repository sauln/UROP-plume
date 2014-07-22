#!/bin/bash

python -m cProfile -o profile/profile $1.py
python profile/profileReader.py

