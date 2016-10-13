#!/bin/bash
g++ -c linear_regression.cpp
g++ -o run1 linear_regression.o
./run1 $1 $2
