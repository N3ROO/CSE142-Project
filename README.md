## CSE142 Project

### Setting up your dev env

### Prerequisites:
- Install python 3.7.7 (tensorflow supports python 3.5-3.7) 64 bits to prevent strange errors.
- Install [Microsoft Visual C++ Redistributable packages for x64 (2015, 2017, 2019)](https://aka.ms/vs/16/release/vc_redist.x64.exe).
- Install virtualenv: `python -m pip install virtualenv`


### Dependencies
For linux or macOs, go to the [official doc](https://www.tensorflow.org/install/pip#2-create-a-virtual-environment-recommended)
- `python --version` needs to be Python 3.5-3.7
- `pip3 --version` needs to be >= 19
- `virtualenv --version` should be installed (version used for this project: 20.0.18)

### Virtual environment
Open a terminal in this folder.
- `virtualenv --system-site-packages -p python ./venv` will create a new folder "/venv"
- `.\venv\Scripts\activate` activate the env. Now when you'll use python commands, they will have effect on that environment.
- You can upgrade pip if needed: `pip install --upgrade pip`
- `pip list` shwos the packages installed in the environment
- `deactivate` to exit the virtual environment

### Tensorflow
*Make sure that you are in the virtual environment.*
- Run `pip install --upgrade tensorflow` to install tensorflow. This project works with tensorflow 2.1
- Verify the installation: `python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"`

### Required packages
*Make sure that you are in the virtual environment.*

- Install requirements: `pip install -r requirements.txt`

### Run the code using your GPU (if you've got a good one)

- Follow these [instructions](https://www.tensorflow.org/install/gpu) and install Cuda 10.1 and not 10.0 or 10.2!!
- Instructions on how to install cuDNN because it's not written in the given
website: [doc nvidia](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

### Mess around with the program
*Make sure that you are in the virtual environment.*

**Jupyter notebook:**
- `ipython kernel install --user --name=.venv` install a kernel inside the environment, to use to run in the Jupyter notebook there
- Run jupyter notebook: `jupyter notebook` (**and not `python -m jupyter notebook`!**)
- Open `gan.ipynb`. Make sure that the kernel is selected when you open it (**.venv**). To change it, go to "Kernel", and then "Change kernel".

**Google collab**
- Simply import the .ipynb file

### Data

*the data comes from this [source](http://yann.lecun.com/exdb/mnist/)*