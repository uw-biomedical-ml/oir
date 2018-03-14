# OIR segmentation

This is the code to train a CNN to do the segmentation of the vaso-obliteration zone and the neovascular tufts/clusters in the images obtained from the mouse model of oxygen-induced retinopathy (OIR).

## Setup
All code is implemented in [Torch](http://torch.ch/).

First [install Torch](http://torch.ch/docs/getting-started.html#installing-torch), then
update / install the following packages:

```bash
luarocks install torch
luarocks install nn
luarocks install nngraph
luarocks install image
luarocks install gnuplot
```

### (Optional) GPU Acceleration

If you have an NVIDIA GPU, you can accelerate all operations with CUDA.

```bash
luarocks install cutorch
luarocks install cunn
```

### .tif, .tiff support
The code supports .tif and .tiff format, but you need to install the following package:

```bash
apt-get install libgraphicsmagick1-dev
luarocks install graphicsmagick
```

## Run segmentation on new image

### Single image
```bash
th predict.lua --imageFile 'image/raw.png' --outputdir 'output'
```

Here is an example of the input and output
<div align='center'>
  <img src='sample/raw.png' height='350px'>
  <img src='sample/predict.png' height="350px">
</div>

### Batch images
```bash
th predict_batch.lua --imageFolder 'sample/batch' --outputdir 'output'
```
The program will process all the images in the given "imageFolder", including all the subfolders. The folder structure will be copied to the given "outputdir", and the segmenation results will be saved in a "result" folder in each corressponding directories.
