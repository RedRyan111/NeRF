# NeRF

An unofficial Pytorch implementation of the paper

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  

This code is based on the original code by [krrish94](https://github.com/krrish94/nerf-pytorch) and another PyTorch implementation by [bmild](https://github.com/bmild/nerf)

## Running the experiment:
For the tiny nerf data, download the 'tiny_nerf_data.zip' and unzip the folder in the data file. <br>
For custom experiments, put your data in the data folder and then you can edit the dataloaders in the dataloader folders, or create your own. <br>
As seen in main.py, the dataloader only needs to provide the rest of the code with the image and camera to world transformation matrix. 

## Final Training Results:

![Final Training Results](https://github.com/RedRyan111/Tiny_NeRF/blob/main/results/Training_Result.png)
