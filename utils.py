import torch
import torch
import torch.nn as nn
import logging
import numpy as np
import torch.nn.functional as F
import math
import os
import yaml
from ruamel import yaml as ryl
import scipy.io as sio
import datetime


def setup_runtime(args, saveyaml=False):
    """Load configs."""
    # Setup CUDA
    cuda_device_id = args.gpu   # 0  1    0,1   cpu
    # get the device id to be a list including '0','1' ....
    if len(cuda_device_id[0]) > 1:
        if 'c' in cuda_device_id[0]:
            cuda_device_id='cpu'
        elif ',' in cuda_device_id[0]:
            result = cuda_device_id[0].split(',')
            cuda_device_id = result

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    configs = {}
    if args.config is not None and os.path.isfile(args.config):
        configs = load_yaml(args.config)
    else:
        configs['config'] = args.config
        configs['seed'] = args.seed
        configs['num_workers'] = args.num_workers
        # the device setting
        if torch.cuda.is_available() and 'cpu' not in cuda_device_id and cuda_device_id[0] !='None':
            device_list = []
            for item in cuda_device_id:
                if item:
                    devicetemp = 'cuda:' + str(item)
                    device_list.append(devicetemp)
            configs['device_list'] = device_list
        else:
            configs['device_list'] = ['cpu']
        configs['num_epoch'] = args.num_epoch
        configs['batch_size'] = args.batch_size
        configs['lr'] = args.lr
        if args.main_path is not None:
            configs['main_path'] = args.main_path
        if args.weight_load is not None:
            configs['weight_load'] = args.weight_load
        if args.weight_save is not None:
            configs['weight_save'] = args.weight_save
        if args.log_path is not None:
            configs['log_path'] = args.log_path
        configs['data_num'] = args.data_num
        configs['save_flag'] = args.save_flag
        configs['load_flag'] = args.load_flag
        configs['version'] = args.version
        configs['dataset'] = args.dataset
        configs['train'] = args.train
        configs['test'] = args.test

        # check params
        if args.data_num < args.batch_size:
            raise ValueError(f"the parameter 'data_num' need to larger than the batch size:{args.batch_size}")
        if args.data_num < args.batch_size:
            raise ValueError(f"the data number need to bigger than the batch size :{args.batch_size}")

    curpath = os.path.dirname(os.path.realpath(__file__))
    yamlpath = os.path.join(curpath, "configs.yaml")

    if saveyaml:
        save_yaml(yamlpath, configs)
    print(f"Environment: GPU {cuda_device_id} ")

    return configs


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(path,dict):
    print(f"Saving configs on {path}")
    with open(path, "w", encoding="utf-8") as f:
        ryl.dump(dict, f, Dumper=ryl.RoundTripDumper)


def print_c(*args, color='cyan'):
    """  print with color  """
    strings = args
    # print(strings)
    string0 = ' '
    for str_temp in strings:
        string0 = string0 + str(str_temp)
    string_sum = string0
    if color == 'cyan':
        print("\033[1;36m" + string_sum + "\033[0m")
    elif color == 'cyans':
        print("\033[1;36m" + string_sum + "\033[0m")
        logging.info(string_sum)
    elif color == 'pink':
        print("\033[1;36m" + string_sum + "\033[0m")
    elif color == 'white':
        print("\033[1;29m" + string_sum + "\033[0m")
    elif color == 'black':
        print("\033[1;30m" + string_sum + "\033[0m")
    elif color == 'red':
        print("\033[1;31m" + string_sum + "\033[0m")
    elif color == 'reds':
        print("\033[1;31m" + string_sum + "\033[0m")
        logging.info(string_sum)
    elif color == 'green':
        print("\033[1;32m" + string_sum + "\033[0m")
    elif color == 'greens':
        print("\033[1;32m" + string_sum + "\033[0m")
        logging.info(string_sum)
    elif color == 'yellow':
        print("\033[1;33m" + string_sum + "\033[0m")
    elif color == 'yellows':
        print("\033[1;33m" + string_sum + "\033[0m")
        logging.info(string_sum)
    elif color == 'blue':
        print("\033[1;34m" + string_sum + "\033[0m")
    elif color == 'purple':
        print("\033[1;35m" + string_sum + "\033[0m")
    elif color == 'blues':
        print("\033[1;34m" + string_sum + "\033[0m")
        logging.info(string_sum)
    else:
        print(string_sum)


def save_logfile(str1=None,str2=None,flag=0,filename=None):
    """
    model_save the log for the traing and test
    the filename need to be the absolute path
    str1 for the string print when flag==0
    str2 for the string print when flag!=0
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S',
                        filename=filename,
                        filemode='a'
                        )
    if flag == 1:
        logging.info(str1)
    logging.info(str2)


def get_param_num(model, detail_flag=0):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    for name, param in model.named_parameters():
        mulValue = np.prod(param.size())
        if detail_flag:
            print(name,':', mulValue)
        Total_params += mulValue
        if param.requires_grad:
            Trainable_params += mulValue
        else:
            print('NonTrainable_params',name)
            NonTrainable_params += mulValue
    print('Trainable_params: %.4f M' % (1e-6 * Trainable_params))
    print('NonTrainable_params: %.4f M' % (1e-6 * NonTrainable_params))
    print('Total params: %.4f M' % (1e-6 * Total_params))
    return Total_params


def print_debug(*args, color='cyan'):
    # logging.basicConfig(level=logging.DEBUG)
    strings = args
    string0 = ''
    for str_temp in strings:
        string0 = string0 + str(str_temp)
    string = string0
    if color == 'cyan':
        logging.debug("\033[1;36m" + string + "\033[0m")
    elif color == 'pink':
        logging.debug(string)
    elif color == 'white':
        logging.debug("\033[1;29m" + string + "\033[0m")
    elif color == 'black':
        logging.debug("\033[1;30m" + string + "\033[0m")
    elif color == 'red':
        logging.debug("\033[1;31m" + string + "\033[0m")
    elif color == 'green':
        logging.debug("\033[1;32m" + string + "\033[0m")
    elif color == 'yellow':
        logging.debug("\033[1;33m" + string + "\033[0m")
    elif color == 'blue':
        logging.debug("\033[1;34m" + string + "\033[0m")
    elif color == 'purple':
        logging.debug("\033[1;35m" + string + "\033[0m")
    else:
        logging.debug(string)


def print_info(*args, color='cyan'):
    strings = args
    # print(strings)
    string0 = ''
    for str_temp in strings:
        string0 = string0 + str(str_temp)
    string = string0
    # print(string)
    if color == 'cyan':
        logging.info("\033[1;36m " + string + " \033[0m")
    elif color == 'pink':
        logging.info(string)
    elif color == 'white':
        logging.info("\033[1;29m" + string + "\033[0m")
    elif color == 'black':
        logging.info("\033[1;30m" + string + "\033[0m")
    elif color == 'red':
        logging.info("\033[1;31m" + string + "\033[0m")
    elif color == 'green':
        logging.info("\033[1;32m" + string + "\033[0m")
    elif color == 'yellow':
        logging.info("\033[1;33m" + string + "\033[0m")
    elif color == 'blue':
        logging.info("\033[1;34m" + string + "\033[0m")
    elif color == 'purple':
        logging.info("\033[1;35m" + string + "\033[0m")
    else:
        logging.info(string)


def print_warn(*args, color='cyan'):
    strings = args
    # print(strings)
    string0 = ''
    for str_temp in strings:
        string0 = string0 + str(str_temp)
    string = string0
    # print(string)
    if color == 'cyan':
        logging.warning("\033[1;36m" + string + "\033[0m")
    elif color == 'pink':
        logging.warning(string)
    elif color == 'white':
        logging.warning("\033[1;29m" + string + "\033[0m")
    elif color == 'black':
        logging.warning("\033[1;30m" + string + "\033[0m")
    elif color == 'red':
        logging.info("\033[1;31m" + string + "\033[0m")
    elif color == 'green':
        logging.warning("\033[1;32m" + string + "\033[0m")
    elif color == 'yellow':
        logging.warning("\033[1;33m" + string + "\033[0m")
    elif color == 'blue':
        logging.warning("\033[1;34m" + string + "\033[0m")
    elif color == 'purple':
        logging.warning("\033[1;35m" + string + "\033[0m")
    else:
        logging.warning(string)


def tensor_shift_5D(x, device='cpu'):
    shape_x = x.shape
    # shape_x = shape_x
    device = x.device
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x[2:], 2)).int()  # floor
    half_dim1 = torch.ceil(torch.div(shape_x[2:], 2)).int()   # ceil
    data_rotate[:,:,-half_dim1[0]:, -half_dim1[1]:, -half_dim1[2]:] = x[:, :, 0:half_dim1[0],
                                                                        0:half_dim1[1],
                                                                        0:half_dim1[2]]
    data_rotate[:,:,0:half_dim0[0], -half_dim1[1]:, -half_dim1[2]:] = x[:, :, -half_dim0[0]:,
                                                                        0:half_dim1[1],
                                                                        0:half_dim1[2]]
    data_rotate[:,:,-half_dim1[0]:, 0:half_dim0[1], -half_dim1[2]:] = x[:, :, 0:half_dim1[0],
                                                                        -half_dim0[1]:,
                                                                        0:half_dim1[2]]
    data_rotate[:,:,0:half_dim0[0], 0:half_dim0[1]:, -half_dim1[2]:] = x[:, :, -half_dim0[0]:,
                                                                        -half_dim0[1]:,
                                                                        0:half_dim1[2]]
    data_rotate[:,:,-half_dim1[0]:, -half_dim1[1]:, 0:half_dim0[2]] = x[:, :, 0:half_dim1[0],
                                                                        0:half_dim1[1],
                                                                        -half_dim0[2]:]
    data_rotate[:,:,0:half_dim0[0], -half_dim1[1]:, 0:half_dim0[2]] = x[:, :, -half_dim0[0]:,
                                                                        0:half_dim1[1],
                                                                        -half_dim0[2]:]
    data_rotate[:,:,-half_dim1[0]:, 0:half_dim0[1], 0:half_dim0[2]] = x[:, :, 0:half_dim1[0],
                                                                        -half_dim0[1]:,
                                                                        -half_dim0[2]:]
    data_rotate[:,:,0:half_dim0[0], 0:half_dim0[1], 0:half_dim0[2]] = x[:, :, -half_dim0[0]:,
                                                                        -half_dim0[1]:,
                                                                        -half_dim0[2]:]
    return data_rotate


def tensor_shift_2D(x,device='cpu'):
    ''' fftshift same as the matlab function '''
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x, 2)).int()    # floor
    half_dim1 = torch.ceil(torch.div(shape_x, 2)).int()     # ceil

    # print('half_dim', half_dim0, half_dim1)
    shape_x = shape_x
    # print('shape_x', shape_x)
    # 00
    data_rotate[-half_dim1[1]:, -half_dim1[1]:] = x[0:half_dim1[0], 0:half_dim1[1]]
    # 01
    data_rotate[0:half_dim1[0], -half_dim1[1]:] = x[-half_dim1[0]:, 0:half_dim1[1]]

    # 10
    data_rotate[-half_dim0[0]:, 0:half_dim0[1]] = x[0:half_dim0[0], -half_dim0[1]:]
    # 11
    data_rotate[0:half_dim0[0]:, 0:half_dim0[1]] = x[-half_dim0[0]:, -half_dim0[1]:]

    return data_rotate


def tensor_shift_3D(x,device='cpu'):
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x, 2)).int()   # floor
    half_dim1 = torch.ceil(torch.div(shape_x, 2)).int()    # ceil
    # print(half_dim1)
    # print(half_dim0)
    # 000
    data_rotate[-half_dim1[0]:, -half_dim1[1]:, -half_dim1[2]:] = x[0:half_dim1[0],
                                                                    0:half_dim1[1],
                                                                    0:half_dim1[2]]
    # 100
    data_rotate[0:half_dim0[0], -half_dim1[1]:, -half_dim1[2]:] = x[-half_dim0[0]:,
                                                                    0:half_dim1[1],
                                                                    0:half_dim1[2]]
    # 010
    data_rotate[-half_dim1[0]:, 0:half_dim0[1], -half_dim1[2]:] = x[0:half_dim1[0],
                                                                    -half_dim0[1]:,
                                                                    0:half_dim1[2]]
    # 110
    data_rotate[0:half_dim0[0], 0:half_dim0[1]:, -half_dim1[2]:] = x[-half_dim0[0]:,
                                                                     -half_dim0[1]:,
                                                                     0:half_dim1[2]]

    # 001
    data_rotate[-half_dim1[0]:, -half_dim1[1]:, 0:half_dim0[2]] = x[0:half_dim1[0],
                                                                     0:half_dim1[1],
                                                                     -half_dim0[2]:]
    # 101
    data_rotate[0:half_dim0[0], -half_dim1[1]:, 0:half_dim0[2]] = x[-half_dim0[0]:,
                                                                    0:half_dim1[1],
                                                                    -half_dim0[2]:]
    # 011
    data_rotate[-half_dim1[0]:, 0:half_dim0[1], 0:half_dim0[2]] = x[0:half_dim1[0],
                                                                    -half_dim0[1]:,
                                                                    -half_dim0[2]:]
    # 111
    data_rotate[0:half_dim0[0], 0:half_dim0[1], 0:half_dim0[2]] = x[-half_dim0[0]:,
                                                                    -half_dim0[1]:,
                                                                    -half_dim0[2]:]

    return data_rotate


def tensor_ir_shift_2D(x,device='cpu'):
    ''' fftshift same as the matlab function '''
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x, 2)).int()    # floor
    half_dim1 = torch.ceil(torch.div(shape_x, 2)).int()     # ceil
    # print('half_dim', half_dim0, half_dim1)
    # 00
    data_rotate[0:half_dim1[0], 0:half_dim1[1]] = x[-half_dim1[1]:, -half_dim1[1]:]
    # 01
    data_rotate[-half_dim1[0]:, 0:half_dim1[1]] = x[0:half_dim1[0], -half_dim1[1]:]
    # 10
    data_rotate[0:half_dim0[0], -half_dim0[1]:] = x[-half_dim0[0]:, 0:half_dim0[1]]
    # 11
    data_rotate[-half_dim0[0]:, -half_dim0[1]:] = x[0:half_dim0[0]:, 0:half_dim0[1]]

    return data_rotate


def tensor_ir_shift_3D(x,device='cpu'):
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x, 2)).int()   # floor
    half_dim1 = torch.ceil(torch.div(shape_x, 2)).int()    # ceil
    # 000
    data_rotate[0:half_dim1[0], 0:half_dim1[1], 0:half_dim1[2]] = x[-half_dim1[0]:, -half_dim1[1]:, -half_dim1[2]:]
    # 100
    data_rotate[-half_dim0[0]:,0:half_dim1[1],0:half_dim1[2]] = x[0:half_dim0[0], -half_dim1[1]:, -half_dim1[2]:]
    # 010
    data_rotate[0:half_dim1[0],-half_dim0[1]:,0:half_dim1[2]] = x[-half_dim1[0]:, 0:half_dim0[1], -half_dim1[2]:]
    # 110
    data_rotate[-half_dim0[0]:,-half_dim0[1]:,0:half_dim1[2]] = x[0:half_dim0[0], 0:half_dim0[1]:, -half_dim1[2]:]
    # 001
    data_rotate[0:half_dim1[0],0:half_dim1[1],-half_dim0[2]:] = x[-half_dim1[0]:, -half_dim1[1]:, 0:half_dim0[2]]
    # 101
    data_rotate[-half_dim0[0]:, 0:half_dim1[1], -half_dim0[2]:] = x[0:half_dim0[0], -half_dim1[1]:, 0:half_dim0[2]]
    # 011
    data_rotate[0:half_dim1[0],-half_dim0[1]:,-half_dim0[2]:] = x[-half_dim1[0]:, 0:half_dim0[1], 0:half_dim0[2]]
    # 111
    data_rotate[-half_dim0[0]:,-half_dim0[1]:,-half_dim0[2]:] = x[0:half_dim0[0], 0:half_dim0[1], 0:half_dim0[2]]

    return data_rotate


def tensor_ir_shift_4D(x,device='cpu'):
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x[1:], 2)).int()   # floor
    half_dim1 = torch.ceil(torch.div(shape_x[1:], 2)).int()    # ceil
    # print(half_dim1)
    # print(half_dim0)
    data_rotate[:,0:half_dim1[0], 0:half_dim1[1], 0:half_dim1[2]] = x[:,-half_dim1[0]:, -half_dim1[1]:, -half_dim1[2]:]
    # 100
    data_rotate[:,-half_dim0[0]:,0:half_dim1[1],0:half_dim1[2]] = x[:,0:half_dim0[0], -half_dim1[1]:, -half_dim1[2]:]
    # 010
    data_rotate[:,0:half_dim1[0],-half_dim0[1]:,0:half_dim1[2]] = x[:,-half_dim1[0]:, 0:half_dim0[1], -half_dim1[2]:]
    # 110
    data_rotate[:,-half_dim0[0]:,-half_dim0[1]:,0:half_dim1[2]] = x[:,0:half_dim0[0], 0:half_dim0[1]:, -half_dim1[2]:]
    # 001
    data_rotate[:,0:half_dim1[0],0:half_dim1[1],-half_dim0[2]:] = x[:,-half_dim1[0]:, -half_dim1[1]:, 0:half_dim0[2]]
    # 101
    data_rotate[:,-half_dim0[0]:, 0:half_dim1[1], -half_dim0[2]:] = x[:,0:half_dim0[0], -half_dim1[1]:, 0:half_dim0[2]]
    # 011
    data_rotate[:,0:half_dim1[0],-half_dim0[1]:,-half_dim0[2]:] = x[:,-half_dim1[0]:, 0:half_dim0[1], 0:half_dim0[2]]
    # 111
    data_rotate[:,-half_dim0[0]:,-half_dim0[1]:,-half_dim0[2]:] = x[:,0:half_dim0[0], 0:half_dim0[1], 0:half_dim0[2]]
    return data_rotate


def tensor_ir_shift_5D(x,device='cpu'):
    shape_x = x.shape
    # print(shape_x, x.dtype)
    data_rotate = x.clone()
    shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    half_dim0 = torch.floor(torch.div(shape_x[2:], 2)).int()   # floor
    half_dim1 = torch.ceil(torch.div(shape_x[2:], 2)).int()    # ceil
    # print(half_dim1)
    # print(half_dim0)
    data_rotate[:,:, 0:half_dim1[0], 0:half_dim1[1], 0:half_dim1[2]] = x[:,:,-half_dim1[0]:, -half_dim1[1]:, -half_dim1[2]:]
    # 100
    data_rotate[:,:,-half_dim0[0]:,0:half_dim1[1],0:half_dim1[2]] = x[:,:,0:half_dim0[0], -half_dim1[1]:, -half_dim1[2]:]
    # 010
    data_rotate[:,:,0:half_dim1[0],-half_dim0[1]:,0:half_dim1[2]] = x[:,:,-half_dim1[0]:, 0:half_dim0[1], -half_dim1[2]:]
    # 110
    data_rotate[:,:,-half_dim0[0]:,-half_dim0[1]:,0:half_dim1[2]] = x[:,:,0:half_dim0[0], 0:half_dim0[1]:, -half_dim1[2]:]
    # 001
    data_rotate[:,:,0:half_dim1[0],0:half_dim1[1],-half_dim0[2]:] = x[:,:,-half_dim1[0]:, -half_dim1[1]:, 0:half_dim0[2]]
    # 101
    data_rotate[:,:,-half_dim0[0]:, 0:half_dim1[1], -half_dim0[2]:] = x[:,:,0:half_dim0[0], -half_dim1[1]:, 0:half_dim0[2]]
    # 011
    data_rotate[:,:,0:half_dim1[0],-half_dim0[1]:,-half_dim0[2]:] = x[:,:,-half_dim1[0]:, 0:half_dim0[1], 0:half_dim0[2]]
    # 111
    data_rotate[:,:,-half_dim0[0]:,-half_dim0[1]:,-half_dim0[2]:] = x[:,:,0:half_dim0[0], 0:half_dim0[1], 0:half_dim0[2]]
    return data_rotate


def tensor_unsqueeze(data,dim):
    for item in dim:
        data=torch.unsqueeze(data, item)
    return data


def mirror_data_for_fft(x):
    shape_x = x.shape
    # device = x.device
    data_mirror=x
    flip_x=torch.flip(x[:,:,:,:,0:int(shape_x[4]//2)+1], dims=[4])
    data_mirror[:,:,:,:,0:int(shape_x[4]//2)+1] = flip_x
    return data_mirror


def fftshift_tensor(x, device='cpu'):
    ''' on cpu  the processing would be much slower '''
    shape_x = x.shape
    device = x.device
    # shape_x = torch.tensor(shape_x, dtype=torch.int16, device=device)
    if len(shape_x)==2:
        data_rotate = tensor_shift_2D(x, device)
    elif len(shape_x) == 3:
        # print('The length of the data is 3 ')
        data_rotate = tensor_shift_3D(x, device)
    elif len(shape_x) == 5:
        # print('The length of the data is 5')
        data_rotate = tensor_shift_5D(x, device)
    else:
        print('Unsupported dimension')
    # print('shift done ! ')
    return data_rotate


def ifftshift_tensor(x,device='cpu'):
    """ Same as the fftshift_tensor when the dim of the x is even """
    """ shape [2,224,224,126] time cost 0.0098"""
    shape_x = x.shape
    device = x.device
    if len(shape_x) == 2:
        data_rotate = tensor_ir_shift_2D(x, device)
    elif len(shape_x) == 3:
        # print('The length of the data is 3 ')
        data_rotate = tensor_ir_shift_3D(x, device)
    elif len(shape_x) == 4:
        # print('The length of the data is 5')
        data_rotate = tensor_ir_shift_4D(x, device)
    elif len(shape_x) == 5:
        # print('The length of the data is 5')
        data_rotate = tensor_ir_shift_5D(x, device)
    else:
        print('Unsupported dimension')
    # print('shift done ! ')
    return data_rotate


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.03797616, 0.044863533, 0.03797616],
                  [0.044863533, 0.053, 0.044863533],
                  [0.03797616, 0.044863533, 0.03797616]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x

def filter3d_torch(x, kernel,pad_op):
    x=pad_op(x)
    return F.conv3d(x, kernel)

class LossHFEN_MSE(torch.nn.Module):
    def __init__(self, device='cpu', size=15, sigma=1.5, weight=4e-6):
        super(LossHFEN_MSE, self).__init__()
        self.device = device
        self.weight = weight
        grid = torch.linspace(-int(size // 2), int(size // 2), size)
        z, y, x = torch.meshgrid(grid, grid, grid)
        h = torch.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
        h_sum = torch.sum(h)
        kernel = h / h_sum
        arg = ((x * x + y * y + z * z) / (math.pow(sigma, 4)) - 3 / (math.pow(sigma, 2)))
        H = torch.mul(arg, kernel)
        H = H - torch.sum(H) / (math.pow(size, 3))
        H = torch.unsqueeze(H, 0)
        self.kernel = torch.unsqueeze(H, 0)
        pad_size = int(size // 2)
        self.pad_op = nn.ReplicationPad3d(pad_size)

    def forward(self, img, ref, flag=0, mask=None, print_flag=0):
        ''' compute hfen different with matlab '''
        device = img.device
        kernel = self.kernel.to(device)
        pad_op = self.pad_op.to(device)
        img_f = filter3d_torch(img, kernel, pad_op)
        ref_f = filter3d_torch(ref, kernel, pad_op)
        if flag == 1:
            if mask == None:
                mask = get_mask_from_data(ref)
            img_f = img_f * mask
            ref_f = ref_f * mask
            hfen = cal_rmse_qsm(img_f, ref_f)
        else:
            hfen = cal_rmse_qsm(img_f, ref_f)
        hfen = self.weight*hfen
        mse_loss = torch.mean(torch.pow((img-ref),2))
        if print_flag:
            print('hfen loss:%.3e' % hfen,'mse_loss:%.3e'% mse_loss)
        return mse_loss+hfen


class gdl3d(nn.Module):
    def __init__(self, device):
        super(gdl3d, self).__init__()
        kernel = [[[-1, 1]]]
        kernelx = np.transpose(kernel, [2, 1, 0])  # exchange the x with y
        # print(kernelx.shape)
        kernely = np.transpose(kernel, [1, 2, 0])  # exchange the x with y
        # print(kernely.shape)
        kernelz = np.transpose(kernel, [0, 1, 2])  # exchange the x with z
        # print(kernelz.shape)

        self.kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).to(device)
        self.kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).to(device)
        self.kernelz = torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0).to(device)
        # baseFilter = np.reshape(baseFilter,(1,)+baseFilter.shape+(1,1,)) # get 1x1x2

    def forward(self,gen_frames, gt_frames):
        alpha=1
        # print(gen_frames.device)
        # print(gen_frames.shape) # torch.Size([4, 1, 64, 64, 16])
        # print(gt_frames.shape)  # torch.Size([4, 1, 64, 64, 16])
        gen_dx = F.conv3d(gen_frames, self.kernelx, padding=[1, 0, 0])
        gen_dy = F.conv3d(gen_frames, self.kernely, padding=[0, 1, 0])
        gen_dz = F.conv3d(gen_frames, self.kernelz, padding=[0, 0, 1])
        # print(gen_dx.shape)  # torch.Size([4, 1, 65, 64, 16])
        # print(gen_dy.shape)  # torch.Size([4, 1, 64, 65, 16])
        # print(gen_dz.shape)  # torch.Size([4, 1, 64, 64, 17])
        gt_dx = F.conv3d(gt_frames, self.kernelx, padding=[1,0,0])
        gt_dy = F.conv3d(gt_frames, self.kernely, padding=[0,1,0])
        gt_dz = F.conv3d(gt_frames, self.kernelz, padding=[0,0,1])
        # print(gt_dx.shape)  # torch.Size([4, 1, 65, 64, 16])
        # print(gt_dy.shape)  # torch.Size([4, 1, 64, 65, 16])
        # print(gt_dz.shape)  # torch.Size([4, 1, 64, 64, 17])
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)
        grad_diff_z = torch.abs(gt_dz - gen_dz)
        final_loss = torch.mean((torch.mean(grad_diff_x ** alpha) +
                                 torch.mean(grad_diff_y ** alpha) + torch.mean(grad_diff_z ** alpha)))

        return final_loss


class get_grad3d(nn.Module):
    '''get the abs of the grad'''
    def __init__(self, device):
        super(get_grad3d, self).__init__()
        kernel = [[[-1, 1]]]
        kernelx = np.transpose(kernel, [2, 1, 0])  # exchange the x with y
        # print(kernelx.shape)
        kernely = np.transpose(kernel, [1, 2, 0])  # exchange the x with y
        # print(kernely.shape)
        kernelz = np.transpose(kernel, [0, 1, 2])  # exchange the x with z
        # print(kernelz.shape)
        self.kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).to(device)
        self.kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).to(device)
        self.kernelz = torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0).to(device)
        # baseFilter = np.reshape(baseFilter,(1,)+baseFilter.shape+(1,1,)) # get 1x1x2


    def forward(self, gen_frames):
        alpha=1
        # print(gen_frames.device)
        # print(gen_frames.shape) # torch.Size([4, 1, 64, 64, 16])
        # print(gt_frames.shape)  # torch.Size([4, 1, 64, 64, 16])
        gen_dx = torch.mean(torch.abs(F.conv3d(gen_frames, self.kernelx, padding=[1, 0, 0])))
        gen_dy = torch.mean(torch.abs(F.conv3d(gen_frames, self.kernely, padding=[0, 1, 0])))
        gen_dz = torch.mean(torch.abs(F.conv3d(gen_frames, self.kernelz, padding=[0, 0, 1])))
        # print(gen_dx.shape)  # torch.Size([4, 1, 65, 64, 16])
        # print(gen_dy.shape)  # torch.Size([4, 1, 64, 65, 16])
        # print(gen_dz.shape)  # torch.Size([4, 1, 64, 64, 17])
        # print(gt_dx.shape)  # torch.Size([4, 1, 65, 64, 16])
        # print(gt_dy.shape)  # torch.Size([4, 1, 64, 65, 16])
        # print(gt_dz.shape)  # torch.Size([4, 1, 64, 64, 17])
        final_loss = torch.add(torch.add(gen_dx, gen_dy),gen_dz)

        return final_loss


class Loss_mse(torch.nn.Module):
    def __init__(self, w_grad=0):
        super(Loss_mse, self).__init__()
        self.w_grad = w_grad
        # self.w_l2=w_l2

    def forward(self, pred, coslabel, loss_txt_path=None,print_flag=0):
        l2_loss = torch.mean(torch.pow(torch.abs(pred - coslabel), 2))
        now = datetime.datetime.now()
        time_record = '{}/{}/{},{}:{}:{} '.format(now.year, now.month, now.day,now.hour, now.minute, now.second)
        # if print_flag:
        #     print('L2_loss:%.4e' % l2_loss, 'Total loss:%.4e ' %(l2_loss))
        if loss_txt_path:
            with open(loss_txt_path, 'a') as f:
                str_write = '* l2_loss:%.4e ' % l2_loss
                str_write = '\n'+time_record+'\n'+str_write
                f.write(str_write)
                print('* Write loss txt file completed !')
        return l2_loss


def tensor2mat(data):
    ''' get tensor to numpy'''
    device0=data.device
    if device0=='cpu':
        data_save = np.squeeze(((data).detach().numpy()))
    else:
        data_save = np.squeeze(((data.cpu()).detach().numpy()))

    return data_save


def test_index_generate(slice, step):
    list0 = []
    index2 = 0
    overlap_num = 0
    for i in range(int(slice / step + 1)):
        index1 = index2
        index2 = index1 + step
        if index2 < slice:
            list_temp = [index1, index2]
            list0.append(list_temp)
        elif index2 == slice:
            list_temp = [index1, index2]
            list0.append(list_temp)
            break
        else:
            list_temp = [slice - step, slice]
            list0.append(list_temp)
            overlap_num = index1 - (slice - step)
    return list0, overlap_num


def cal_ssim_2D(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    "input: torch.tensor "
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 11)
    mu2 = filter2(im2, window, 11)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 5) - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 5) - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 5) - mu1_mu2
    # print(torch.mean(sigma1_sq),torch.mean(mu1_sq),C1,C2)
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.mean(ssim_map)


def filter2d_torch(x, kernel,pad_op):
    x=pad_op(x)
    return F.conv2d(x, kernel)


def cal_ssim_torch_2d(img, ref, k1=0.01, k2=0.03, win_size=11, L=1.0, roi_flag=False, roi_mask=None):
    """
        calculate the ssim2d   torch version
        roi_flag: True for only calculate ssim in the roi of the ref
        roi_mask: 1 for valid
    """
    if not img.shape == ref.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(img.shape) != 4 or img.shape[0] != 1 or img.shape[1] != 1:
        raise ValueError(f"Please input the images with 4D shape with 1 channel 1 batch,now receive{img.shape}")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = torch_gauss2d(size=win_size, sigma=1.5)
    device = img.device
    window = window.to(device)
    pad_size = int(win_size // 2)
    pad_op = nn.ReplicationPad3d(pad_size).to(device)
    mu1 = filter2d_torch(img, window,pad_op)
    mu2 = filter2d_torch(ref, window,pad_op)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2d_torch(img * img, window,pad_op) - mu1_sq
    sigma2_sq = filter2d_torch(ref * ref, window,pad_op) - mu2_sq
    sigmal2 = filter2d_torch(img * ref, window,pad_op) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if roi_flag:
        device = ref.device
        if roi_mask == None:
            roi_mask = torch.where(torch.abs(ref) > 0, torch.ones_like(ref, device=device),
                                   torch.zeros_like(ref, device=device))
            temp = torch.where(torch.abs(roi_mask) > 0, ssim_map, torch.zeros_like(ref, device=device))
            ssim_result = torch.sum(temp) / torch.sum(roi_mask)
        else:
            temp = torch.where(torch.abs(roi_mask) > 0, ssim_map, torch.zeros_like(ref, device=device))
            ssim_result = torch.sum(temp) / torch.sum(roi_mask)
    else:
        ssim_result = torch.mean(ssim_map)

    return ssim_result


def cal_ssim_torch_3d(img, ref, k1=0.01, k2=0.03, win_size=11, L=1, roi_flag=False, roi_mask=None):
    """
        calculate the ssim3d   torch version   L:bit depth   k1=0.01, k2=0.03, L=1
        roi_flag: True for only calculate ssim in the roi of the ref
        roi_mask: 1 for valid
    """
    if not img.shape == ref.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(img.shape) != 5 or img.shape[0] != 1 or img.shape[1] != 1:
        raise ValueError(f"Please input the images with 5D shape with 1 channel 1 batch,now receive{img.shape}")
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = torch_gauss3d(size=win_size, sigma=1.5)
    device=img.device
    pad_size=int(win_size//2)
    window = window.to(device)
    pad_op = nn.ReplicationPad3d(pad_size).to(device)
    mu1 = filter3d_torch(img, window, pad_op)
    mu2 = filter3d_torch(ref, window, pad_op)  # filter time 0.5637586116790771
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter3d_torch(img * img, window, pad_op) - mu1_sq
    sigma2_sq = filter3d_torch(ref * ref, window, pad_op) - mu2_sq
    sigmal2 = filter3d_torch(img * ref, window, pad_op) - mu1_mu2
    # print(torch.mean(sigma1_sq),torch.mean(mu1_sq),C1,C2)
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if roi_flag:
        device = ref.device
        if roi_mask == None:
            roi_mask = torch.where(torch.abs(ref) > 0, torch.ones_like(ref, device=device),
                                   torch.zeros_like(ref, device=device))
            temp = torch.where(torch.abs(roi_mask) > 0, ssim_map, torch.zeros_like(ref, device=device))
            ssim_result = torch.sum(temp) / torch.sum(roi_mask)
        else:
            temp = torch.where(torch.abs(roi_mask) > 0, ssim_map, torch.zeros_like(ref, device=device))
            ssim_result = torch.sum(temp) / torch.sum(roi_mask)
    else:
        ssim_result = torch.mean(ssim_map)

    return ssim_result


def cal_ssim_torch(img, ref, k1=0.01, k2=0.03, win_size=11, L=1, roi_flag=False, roi_mask=None):
    if len(img.shape) == 4:
        ssim = cal_ssim_torch_2d(img, ref, k1=k1, k2=k2, win_size=win_size, L=L, roi_flag=roi_flag, roi_mask=roi_mask)
    elif len(img.shape) == 5:
        ssim = cal_ssim_torch_3d(img, ref, k1=k1, k2=k2, win_size=win_size, L=L, roi_flag=roi_flag, roi_mask=roi_mask)
    else:
        print('Unsupported data shape for calculate ssim !!')
        return None
    return ssim


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_mask_from_data(data, thre=0):
    """ the data need to be the tensor """
    mask = torch.where((torch.abs(data)>thre),torch.ones_like(data),torch.zeros_like(data))
    return mask


def cal_mse(x, y, mode=0):
    """
    input data need to be 5D
    calculate metric from tensors and return a tensor on gpu
    """
    shape_x = x.shape
    dim1 = shape_x[-1]
    dim2 = shape_x[-2]
    dim3 = shape_x[-3]
    dim1 = torch.from_numpy(np.array(dim1))
    dim2 = torch.from_numpy(np.array(dim2))
    dim3 = torch.from_numpy(np.array(dim3))
    # print(dim1.numpy(), dim2.numpy(),dim3.numpy())
    A = 1/(dim3*dim2*dim1)
    A = A.clone().detach()
    A.type(torch.FloatTensor)
    # print(A.numpy())
    # result = A*sum((x-y)^2)
    # print(A.device,x.device,y.device)
    # print(A.dtype, x.dtype, y.dtype)
    if mode:
        mask = get_mask_from_data(y)
        A=1/torch.sum(mask)
        result = torch.mul(A, (torch.sum(torch.mul(torch.pow((x - y), 2), mask))))
    else:
        # print(torch.sum(torch.pow((x - y), 2)))
        result = torch.mean(torch.pow((x - y), 2))
    # print(result.shape)
    # print(result.device)
    return result


def cal_rmse_qsm(img, ref):
    img = img.reshape(-1)
    ref = ref.reshape(-1)
    rmse = 100*torch.norm(img-ref, 2)/torch.norm(ref, 2)
    return rmse


def cal_rmse(x, y, mode=0):
    if mode:
        mse = cal_mse(x, y, 1)
    else:
        mse = cal_mse(x, y)
    rmse = torch.sqrt(mse)
    return rmse


def cal_psnr(x, y, mode=0):
    """
    mode 0: calculate the whole img  1:calculate the metric only inside the mask
    """
    mse = cal_mse(x, y, mode)
    # print('mse',mse)
    psnr = 10*torch.log10(1/mse)
    return psnr


def cal_hfen(img, ref,flag=0,mask=None,size=15,sigma=1.5):
    ''' compute hfen different with matlab '''
    grid = torch.linspace(-int(size // 2), int(size // 2), size)
    x, z, y = torch.meshgrid(grid, grid, grid)
    # print('x[0,:,:]', x[0,:,:])
    # print('y[0,:,:]', y[0, :, :])
    # print('z[0,:,:]', z[0, :, :])
    h = torch.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    # print('torch.mean(h) %.6e'% torch.mean(h))
    h_sum = torch.sum(h)
    kernel = h / h_sum
    # print('torch.mean(kernel)%.6e' % torch.mean(kernel))
    arg = ((x*x+y*y+z*z)/(math.pow(sigma, 4)) - 3/(math.pow(sigma, 2)))
    # print('torch.mean(arg)%.6e' % torch.mean(arg))
    H = arg*kernel
    arg_save = np.squeeze(arg.numpy())
    kernel_save = np.squeeze(kernel.numpy())
    H_save = np.squeeze(H.numpy())
    dict_save = {'arg_save':arg_save,'H':H_save,'kernel_save':kernel_save}
    sio.savemat('arg_save.mat',dict_save)
    # print('torch.mean(H) %.6e' % torch.mean(H))
    H = H - torch.sum(H) / (math.pow(size, 3))
    # print('torch.mean(H)%.6e' % torch.mean(H))
    H = torch.unsqueeze(H, 0)
    kernel = torch.unsqueeze(H, 0)
    device = img.device
    pad_size = int(size // 2)
    pad_op = nn.ReplicationPad3d(pad_size).to(device)
    kernel = kernel.to(device)
    img_f = filter3d_torch(img, kernel, pad_op)
    ref_f = filter3d_torch(ref, kernel, pad_op)
    # print('img_f:', torch.mean(img_f))
    # print('ref_f:', torch.mean(ref_f))
    # print('img_f.shape', img_f.shape)
    # print('ref_f.shape', ref_f.shape)
    if flag == 1:
        if mask == None:
            mask = get_mask_from_data(ref)
        img_f = img_f*mask
        ref_f = ref_f*mask
        hfen = cal_rmse_qsm(img_f, ref_f)
    else:
        hfen = cal_rmse_qsm(img_f, ref_f)
    return hfen


def cal_4metrics(img, ref):
    """
        calculate metric, return tensor
    """
    if not img.shape == ref.shape:
        raise ValueError("Input Images must have the same dimensions")
    mask = get_mask_from_data(ref)
    img = img * mask
    rmse = cal_rmse_qsm(img, ref)
    hfen = cal_hfen(img, ref, 1)
    psnr = cal_psnr(img, ref, 1)
    ssim = cal_ssim_torch(img, ref)
    return psnr, rmse, ssim, hfen


def conv_RelicationPad3D(data, pd_size):
    ReplicationPad = nn.ReplicationPad3d(padding=(pd_size, pd_size, pd_size, pd_size,pd_size,pd_size))
    data1 = ReplicationPad(data)
    return data1


def matlab_style_gauss2D(shape=(11, 11), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]  # get the
    y,x = np.ogrid[-m:m+1, -n:n+1]       #
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def torch_gauss2d(size=11, sigma=1.5):
    ''' get 2D gauss kernel torch version  dim:1*1*size*size '''
    grid=torch.linspace(-int(size//2),int(size//2),size)
    y,x=torch.meshgrid(grid,grid)
    h = torch.exp(-(x*x+y*y)/(2*sigma*sigma))
    h_sum=torch.sum(h)
    kernel=h/h_sum
    kernel = torch.unsqueeze(kernel,0)
    kernel = torch.unsqueeze(kernel,0)
    return kernel

def torch_gauss3d(size=11, sigma=1.5):
    ''' get 3D gauss kernel torch version  dim:1*1*size*size*size  '''
    grid=torch.linspace(-int(size//2),int(size//2),size)
    z,y,x=torch.meshgrid(grid,grid,grid)
    h = torch.exp(-(x*x+y*y+z*z)/(2*sigma*sigma))
    h_sum=torch.sum(h)
    kernel=h/h_sum
    kernel = torch.unsqueeze(kernel,0)
    kernel = torch.unsqueeze(kernel,0)
    return kernel


def gaussian1D(window_size, sigma):
    gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian1D(window_size, 1.5).unsqueeze(1)
    return 0

def gaussian3d(size,sigma=1.5):
    guasskernel=np.zeros((2*size+1,2*size+1,2*size+1))
    xx = 1/2*np.linspace(-size, size,  2*size+1)
    yy = 1/2*np.linspace(-size, size,  2*size+1)
    zz = 1/2*np.linspace(-size, size,  2*size+1)
    XX,YY,ZZ=np.meshgrid(xx,yy,zz)
    guasskernel = 1/(2*np.pi*sigma)*np.exp(-(np.power(XX, 2)+np.power(YY,2)+np.power(ZZ,2))/(2*np.power(sigma,2)))
    guasskernel0=guasskernel
    # print(guasskernel0[4:8,4:8,4:8])
    return guasskernel0


def filter2(x, kernel, mode='same'):
    kernel_tensor=torch.from_numpy(np.rot90(kernel, 2))
    return F.conv2d(x, kernel_tensor)


def filter3(x, kernel, mode='same'):
    kernel_tensor=torch.from_numpy(kernel)
    return F.conv3d(x, kernel_tensor)


def get_montage_dim(dim1,dim2,dim3):
    ''' calculate the montage dim '''
    mul=dim1*dim2*dim3
    row_temp=np.sqrt(mul)/dim1
    col_temp=np.sqrt(mul)/dim2
    row_temp=np.floor(row_temp)
    col_temp=np.ceil(col_temp)
    if row_temp*col_temp>=dim3:
        row = row_temp
        col = col_temp
    elif (row_temp+1)*col_temp>=dim3:
        row = row_temp+1
        col = col_temp
    else:
        row = row_temp + 1
        col = col_temp + 1
    return int(row), int(col)





