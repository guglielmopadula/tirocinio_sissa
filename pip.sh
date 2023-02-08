#/bin/bash
python3.9 -m venv ../sissa_venv
source ../sissa_venv/bin/activate
pip install --upgrade pip
pip --default-timeout=1000 install -r pip.txt
