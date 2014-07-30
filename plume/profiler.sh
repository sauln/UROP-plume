#!/bin/bash


echo "run sim for a bit"
python -m cProfile -o profiles/profile $1.py $2



echo "totes finished bra"
python profiles/profileReader.py

