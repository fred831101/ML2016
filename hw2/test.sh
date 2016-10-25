#!/bin/bash
g++ -c deep_test.cpp
g++ -o run2 deep_test.o
./run2 $1 $2 $3
