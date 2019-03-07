#!/bin/sh
python3 -m venv asmpcenv
source asmpcenv/bin/activate
pip install numpy pandas statsmodels scipy
git clone git@github.com:degregat/secret-sharing.git
cd secret-sharing
python3 setup.py install
