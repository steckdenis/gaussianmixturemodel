# Gaussian Mixture Model

A Gaussian Mixture Model allows to approximate a function. Given input-output samples, the model identifies the structure of the input and builds knowledge that allows it to predict the value of new points.

This model clusters input points and associates an output value to each cluster. The value predicted for any point is the weighted sum of the values of all the clusters: the closer a cluster is to the input point, the more weight it has.

The model tries to be clever with the clusters and detects when to add or remove ones. Each cluster has a covariance matrix that allows it to span an "oval" region of the input space.

## Installation

Installing the C++ library is easy and uses CMake:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make
sudo make install
```

If you plan to use the Python bindings, it is currently required that you install the library in /usr (the Python bindings look for the library in `sys.lib`)

## Python bindings

This repository contains a `gaussianmixturemodel.py` file. This file exposes the `GaussianMixture` class to Python using ctypes and direct calls to `libgaussianmixturemodel.so`. To use this file, copy it into your project (or in your Python package directory, there is no `setup.py` file yet), then use it like this:

```python
import gaussianmixturemodel as gm
import numpy as np

gmm = gm.GaussianMixture(3, 3, 0.1, 0.1)

gmm.setValue([1, 1, 1], [3, 2, 1])
gmm.setValue([0, 0, 0], [1, 2, 3])

print(gmm.value([.5, .5, .5]))
```

## Links

* A mathematical description of the model is available in Chapter 4 of [my master thesis](http://steckdenis.be/files/master_thesis.pdf).