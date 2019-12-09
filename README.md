# Feature Partitioning for Multi-Task Architectures

This framework can be used to reproduce all experiments performed in _“Feature
Partitioning for Efficient Multi-Task Architectures”_. It provides a variety of
functionality for performing and managing deep learning experiments. In
particular, it helps manage meta-optimization which is useful
when doing hyper-parameter tuning and architecture search.

_Please note, this is not an official Google product._

## Cloud Instance Setup

Start a new Cloud Instance from _“Deep Learning Image: PyTorch
1.0.0”_. Almost everything needed to get the code up and running is
included automatically with the Deep Learning Image.

Following instructions assume the git repository has been pulled and placed in the home directory.

#### Data setup

Download and set up Visual Decathlon data and annotations:

```
wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-devkit.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/share/decathlon-1.0-data.tar.gz
tar zxf decathlon-1.0-devkit.tar.gz
mv decathlon-1.0 ~/mtl/data/decathlon
tar zxf decathlon-1.0-data.tar.gz -C ~/mtl/data/decathlon/data
cd ~/mtl/data/decathlon/data
for f in *.tar; do tar xf "$f"; done
```

ImageNet data must be set up separately, please check out [http://image-net.org/download-images](http://image-net.org/download-images).

#### Code setup

Add the following lines to your ~/.bashrc file:
```
export PATH=~/.local/bin:$PATH
export PYTHONPATH=~/mtl:$PYTHONPATH
ulimit -n 2048
```

Then run the following:
```
source ~/.bashrc
pip install --upgrade torch torchvision tensorflow
```

All code was tested with Python 3.7 and PyTorch 1.0.


## Network Training

A variety of configuration files are available to run different training procedures tested in the paper, some examples include:

```
python main.py -e single_task_network --config exp.dec --model resnet --task_choice 0
python main.py -e partitioned_mtl_network --config exp.dec --task_choice 1-2-3-4
python main.py -e distillation_test --config exp.dist
python main.py -e es_optimization_test --config exp.es_dist
```

The argument ```-e``` indicates the experiment name, and ```--config``` specifies the appropriate configuration file. Further details about network training can be found [here]().
