# Pytorch CNN Visualization
Several CNN visualization implemented in Pytorch

## Prerequisites:
- python3
- virtualenv

## Installation:
In command line `cd` to root directory of the program, run

```
#create a virtual environment with python3
virtualenv -p python3 .env --no-site-packages

#activate the virtual environment
source .env/bin/activate

#install pytorch, guide available at http://pytorch.org/
#in my case, my system is macOS, python 3.6, no cuda, hence I run the following command
pip install http://download.pytorch.org/whl/torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl
pip install torchvision

#install other dependency
pip install matplotlib

#run the program
python visualize.py path/to/image

#top 3 prediction with visualization will be shown
```
