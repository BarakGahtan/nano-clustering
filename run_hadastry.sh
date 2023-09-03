#!/bin/bash

#python hadastry2.py --ksize=5 > output256_5 2>&1 &
#python hadastry2.py --ksize=10 > output256_10 2>&1 &
#python hadastry2.py --ksize=20 > output256_20 2>&1 &
#python hadastry2.py --ksize=30 > output256_30 2>&1 &
#python hadastry2.py --ksize=40 > output256_40 2>&1 &
python hadastry2.py --ksize=50 --levels=2 --nhids=32 > output256_50k_2l_32hid 2>&1 &
python hadastry2.py --ksize=50 --levels=2 --nhids=64 > output256_50k_2l_64hid 2>&1 &
#python hadastry2.py --ksize=100 --levels=8 --nhids=25 > output256_100k_8l_25hid 2>&1 &
#python hadastry2.py --ksize=50 --levels=4 --nhids=64 > output256_