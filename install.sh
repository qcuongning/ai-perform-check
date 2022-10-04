#!/bin/bash#

wget  -O onnx190.whl https://nvidia.box.com/shared/static/w3dezb26wog78rwm2yf2yhh578r5l144.whl

pip3 install onnx190.whl

wget -O torch110.whl https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl

pip3 install torch110.whl
