# NeRF

An unofficial Pytorch implementation of the paper

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution
 
This code is based on the original code by [krrish94](https://github.com/krrish94/nerf-pytorch) and another PyTorch implementation by [bmild](https://github.com/bmild/nerf)

## Running the experiment:
For the tiny nerf data, download the 'tiny_nerf_data.zip' and unzip the folder in the data file. 
For custom experiments, put your data in the data folder and then you can edit the dataloaders in the dataloader folders, or create your own. 
As seen in main.py, the dataloader only needs to provide the rest of the code with the image and camera to world transformation matrix. 

## Final Training Results:

![Final Training Results](https://github.com/RedRyan111/Tiny_NeRF/blob/main/results/Training_Result.png)
