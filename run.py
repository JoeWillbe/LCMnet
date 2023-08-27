import argparse
import torch
from Trainer import Trainer
from utils import setup_runtime
from LCMnet import LCMnet
torch.backends.cudnn.benchmark = True

''' 
python run.py  -gpu 0 -log_path './log_file/LCMnet_xxx.log'  -lr 0.0001 -num_epoch 30 -data_num 6860 -batch_size 8
'''

parser = argparse.ArgumentParser(description='Training configurations.')
parser.add_argument('--config', default=None, type=str, help='Specify a config file path')
parser.add_argument('-gpu', default='cpu', nargs='*', help='Specify a GPU device[default:"0"],setting more than one GPUs '
                                                         'eg, -gpu 0 1')
parser.add_argument('-num_workers', default=4, type=int, help='Specify the number of worker threads for data loaders')
parser.add_argument('-seed', default=0, type=int, help='Specify a random seed')
parser.add_argument('-num_epoch', default=30, type=int, help='Specify the number of epoch[default:20]')
parser.add_argument('-batch_size', default=8, type=int, help='Specify the batch_size[default:4]')
parser.add_argument('-lr', default=1e-4, type=float, help='Specify the learning_rate[default:1e-4]')
parser.add_argument('-version', default='default', type=str, help='Specify the name of model version')
parser.add_argument('-dataset', default='D1', type=str, help='Specify the name of dataset')
parser.add_argument('-train', action='store_true', default=False,  help='Specify the name of dataset')
parser.add_argument('-test', action='store_true', default=False,  help='Specify the name of dataset')
parser.add_argument('-data_num', default=6860, type=int, help='Specify the number of training data number [default:18000]')
parser.add_argument('-save_flag', default=1, type=int, help='Specify the weight saving flag(0/1:default), 1 for save')
parser.add_argument('-load_flag', default=1, type=int, help='Specify the weight loading flag(0/1:default), 1 for load')
parser.add_argument('-main_path', default=None, type=str, help='Specify the path of data.Under the path,the '
                                                               'subfloder "/Train" and "/Validation_full",'
                                                               '"/Test_full" is needed')
parser.add_argument('-weight_load', default=None, type=str, help='Specify the name of model weights for loading')
parser.add_argument('-weight_save', default=None, type=str, help='Specify the name of model weights for saving ')
parser.add_argument('-log_path', default=None, type=str, help='Specify the path of log file,default to output'
                                                              'on the terminal, eg, "/xxx/xxx/xxx/xxx.log" or "xxx.log"')
args = parser.parse_args()


if __name__ == "__main__":
    # *************************   Training Setting   **********************************
    model = LCMnet()
    # args.weight_load = 'model_save/LCMnet_0928'
    # args.weight_save = 'model_save/LCMnet_0928'
    # args.version = 'LCMnet'
    arg_set = setup_runtime(args)
    print(arg_set)
    # *************************************************************************************
    trainer = Trainer(arg_set, model)           # the second input need to be a class
    run_train = arg_set.get('train')     # train on the block data
    run_test = arg_set.get('test')       # test on full size data
    # run_test = True

    if run_train:
        trainer.train_ppm()          # for LCMnet training

    if run_test:
        txt_path = 'Data_Path_Collection.txt'
        trainer.test(txt_path, save_result=1, flag_mode=2, Mag_flag=1, mask_flag=1)




