import model
from imp import reload
reload(model)
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import load



data = load.load_data("cifar10")

X = data[0]
img_depth = data[4]
img_x = data[5]

import Plots
print(X.shape)
Plots.plot_testimg(X[:10, :])

batch_size = 32

num_f1 = 50
num_f2 = 50
num_f3 = 50
f1 = (num_f1, img_depth, 11, 11)
f2 = (num_f2, num_f1, 2, 2)
f3 = (num_f3, num_f2, 3, 3)
filters = [f1]
image_shape = (batch_size, img_depth, img_x, img_x)
CONV = model.Meta_ConvNet(n_epochs=100,
                           batch_size=batch_size,
                           learning_rate=0.005,
                           momentum=0.9,
                           filters=filters,
                           image_shape=image_shape,
                           poolsize=(2, 2),
                           denoising=0.0,
                           dropout=0.5,
                           tied_weights=False,
                           pooling=False,
                           loss="MSE")

CONV.setup()
CONV.fit(X)
costs = CONV.costs

generated = CONV.generate()
print(generated.shape)


