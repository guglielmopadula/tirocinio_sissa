# SISSA Internship working directory
This repo contains the file I use for my intership at SISSA.
Description of the folders:
- OTM: contains some script for computing an Optimal Transport Map (semidiscrete and discrete, and 2D and 3D) that preserves the volume in intermediate times.
- Paper: contains some papers (hoping that I am not violating some copyright rules)
- Poster: contains the poster for the Sissa Summer School
- Blitz: contains the presentation for the Sissa Summer School
- DeepLearning: contains an implementation of a Variational Autoencoder for generating paralallelepipeds and naval hull bulbs of costant volume

To install the packages
```
sudo apt install -y python3-venv
python3.9 -m venv sissa_venv
source sissa_venv/bin/activate
pip install -r pip.txt --extra-index-url https://download.pytorch.org/whl/cu116
```
