#!/bin/bash

cython $1.pyx


gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -I/usr/include/python2.7 -o $1.so $1.c

cp step.so ../step.so



