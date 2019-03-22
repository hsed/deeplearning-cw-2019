
'''
cat /dev/zero | ssh-keygen -q -N ""
cat ~/.ssh/id_rsa.pub
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

# <after adding key to git>
git clone git@github.com:hsed/dl-cw.git

tmux bind m set mouse on
tmux bind M set mouse off

cd dl-cw
python starter.py
'''

### compatible with ufoym/deepmo and tensorflow/nightly-gpu-py3
### warning this script will modify system

import os
from warnings import warn


## print os version or details

#### warnings ####
prompt = input("\n\nWarning: This script will attempt to modify system by \
installing packages and would require sudo permissions to do so, \
do you wish to continue? (Y/N)")

if str(prompt).capitalize() == 'N':
    quit("Script execution cancelled by user!")
elif str(prompt).capitalize() == 'N':
    quit("Unknown choice '%s', cancelling script execution..." % str(prompt))


os.system("pip --version")
os.system("python --version")

os.system('apt update && apt install -y libsm6 libxext6 libxrender-dev zip unzip curl nano')

# keras may or may not exist so just to make sure...
os.system("pip install keras gputil psutil humanize matplotlib opencv-python tqdm \
joblib pandas dill tabulate jupyterlab pillow seaborn")
os.system('pip install git+https://www.github.com/keras-team/keras-contrib.git')

from keras.backend.tensorflow_backend import _get_available_gpus
if len(_get_available_gpus()) == 0:
    warn("No GPU was found!")







############# downloader ###############

import psutil
import humanize
import os
import GPUtil as GPU
GPUs = GPU.getGPUs()
# Colab only provides one GPU and it is not always guaranteed
gpu = GPUs[0]

### need RAM should be around 12.9 GB, which is enough to load the datasets in memory.
### Also, usually, we have available 11.4 GB of GPU memory, which is more than enough to run this code.


def printm():
	process = psutil.Process(os.getpid())
	print("RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
		" | Proc size: " + humanize.naturalsize(process.memory_info().rss))
	print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB"
		.format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))




if __name__ == "__main__":
	printm()

	if not os.path.isfile('hpatches_data.zip'):
		os.system('wget -O hpatches_data.zip \
		https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip')
    

	os.system('unzip -q ./hpatches_data.zip')
	os.system('rm ./hpatches_data.zip')