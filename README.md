# SRCNN
Image super resolution using Dong et al. SRCNN

This repository contains a Python implemenation of Dong, Loy, He and Tang's 'Super Resolution Convolutional Neural Network'. The model takes as input an image and blurs it using bicubic interpolation. The input to the neural network is the blurred image and the CNN is then trained using the high-resolution image as the ground truth.

The output of the SRCNN is a higher resolution image than the blurred image. Examples of a blurred image and outputs of the SRCNN are available in the images folder.

The original MATLAB implementation by Dong et al.: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html <br />
Their paper can be found here: https://arxiv.org/pdf/1501.00092.pdf
