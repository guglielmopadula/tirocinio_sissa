#/bin/bash
pip cache purge
python3.9 -m venv ../sissa_venv
source ../sissa_venv/bin/activate
pip install wheel
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
a=$(python -c "import torch; print(torch.__version__)")
b=$(python -c "import torch; print(torch.version.cuda)")
pip install torch-scatter -f https://data.pyg.org/whl/torch-$a+$b.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-$a+$b.html
pip install torch-geometric
pip install pytorch-lightning
pip install matplotlib pyevtk numpy meshio scipy sympy scikit-learn ordered_set POT torchvision torchaudio spyder
