#!/bin/bash
g++ -c logistic_regression.cpp
g++ -o run1 logistic_regression.o
./run1 $1 $2
