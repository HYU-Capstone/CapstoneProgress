#!/usr/bin/bash

pip3 install netifaces
pip3 install tensorflow-gpu
pip3 install opencv-python
pip3 install boto3
pip3 install numpy scipy matplotlib ipython jupyter pandas sympy nose

curl https://gist.githubusercontent.com/kyujin-cho/f6f87a87b6b153e59b3b3f237e32a662/raw/51584fa03ed5a0b9b90ed2c0dc1362d81ab5e71f/gistfile1.txt > train.py
python3 train.py