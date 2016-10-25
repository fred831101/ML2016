#!/bin/bash
g++ -c logistic_test.cpp
g++ -o run2 logistic_test.o
./run2 $1 $2 $3
