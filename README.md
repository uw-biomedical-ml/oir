# OIR segmentation

This is the code to train a CNN to do the segmentation of the vaso-obliteration zone and the neovascular tufts/clusters in the images obtained from the mouse model of oxygen-induced retinopathy (OIR).

## Setup
All code is implemented in [Torch](http://torch.ch/).

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:

```bash
luarocks install torch
luarocks install nn
luarocks install gnngraph
luarocks install image
luarocks install gnuplot
```

### (Optional) GPU Acceleration

If you have an NVIDIA GPU, you can accelerate all operations with CUDA.

### .tif, .tiff support
The code supports .tif and .tiff format, but you need to install the following package:

```bash
luarocks install graphicsmagick
```
I encountered an `libGraphicsMagickWand.so not found` error when I tried to include the library after intalling using luarocks on some machines. This [link](https://github.com/eladhoffer/ImageNet-Training/issues/5) may help.

```bash
luarocks install cutorch
luarocks install cunn
```
## Run segmentation on new image

```bash
th predict.lua --imageFile 'image/raw.png' --outputdir 'output'
```

Here is an example of the input and output from the model
<div align='center'>
  <img src='sample/raw.png' height='350px'>
  <img src='sample/predict.png' height="350px">
</div>
