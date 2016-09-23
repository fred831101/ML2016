#!/bin/bash
g++ -c test.cpp
g++ -o run1 test.o
./run1 $1 $2