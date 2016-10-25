#!/bin/bash
g++ -c deep_twolayer.cpp
g++ -o run1 deep_twolayer.o
./run1 $1 $2
