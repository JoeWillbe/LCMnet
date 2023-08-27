# LCMnet
Source code for the paper "A Latent Code Based Multi-variable Modulation Network for Susceptibility Mapping‘’
## A Latent Code Based Multi-variable Modulation Network for Susceptibility Mapping
The implementation of LCMnet for QSM reconstruction network  <br />
This code was built and tested on Centos 7

## Overall Framework

 <div align="center"> <img src=Framework.png width = 400 height = 225 /> </div>
Fig.1 The architecture of our proposed LCMnet network. (a)The main framework includes the modulated convolution module,
cross-fusion module, encoder and decoder blocks. (b) The detailed constitutions of some blocks.

## Python Dependencies
numpy
torch=1.7.0
tensorboardX
yaml
scipy
glob


# run training
``` 
python run.py  -gpu 0 -log_path './log_file/LCMnet_xxx.log' -train -lr 0.0001 -num_epoch 30 -data_num 6860 -batch_size 8 -weight_save LCMnet.pkl
```
# run testing
``` 
python run.py  -gpu 0 -log_path './log_file/LCMnet_xxx.log' -test -weight_load LCMnet.pkl
```












