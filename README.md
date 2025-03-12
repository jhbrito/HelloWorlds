# Hello Worlds
Introductory examples of Artificial Inteligence, Machine Learning and Computer Vision.

Topics so far:
- Python
- Numpy
- Matplotlib
- OpenCV
- Qt
- Scikit-learn
- Deep Learning
- Tensorflow
- Satellite images

## Create conda environment
```
conda create -n DevEnv310 python=3.10
conda activate DevEnv310
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"
```

## Import conda environment
```
conda env create -n Project_Environment --file environment.yml
```

## Export conda environment
conda env export > environment.yml

## Export pip environment
pip freeze > requirements.txt

## Import pip environment
pip install -r requirements.txt

## Unofficial Windows Binaries for Python Extension Packages

<https://www.lfd.uci.edu/~gohlke/pythonlibs/>

GDAL 3.1.4 binary: https://download.lfd.uci.edu/pythonlibs/w4tscw6k/cp36/GDAL-3.1.4-cp36-cp36m-win_amd64.whl

rasterio 1.1.8 binary: https://download.lfd.uci.edu/pythonlibs/w4tscw6k/cp36/rasterio-1.1.8-cp36-cp36m-win_amd64.whl

pip install GDAL-3.1.4-cp36-cp36m-win_amd64.whl

pip install rasterio-1.1.8-cp36-cp36m-win_amd64.whl
