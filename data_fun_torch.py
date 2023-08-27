import scipy.io as sio
import numpy as np
import torch
import torch.utils.data as Data
import re
import random
import glob
from utils import print_c


def load_block_image(path, key_name='block_data', numpy_flag=0, device='cpu', dim=5, transpose_flag=0):
    """
    for the Dataset, the output  dim would be the 4D
    the path need to include the mat_file
    0.015-0.018s per data
    """
    image = sio.loadmat(path)
    image = image[key_name]

    if transpose_flag:
        ''' for multi-echo data '''
        # print(image.shape, '----------------')
        image = np.transpose(image, (3, 0, 1, 2))   # make the echo dim to be the first
        image = image.astype('float32')
        # print(image.shape, '*'*13)
        if dim == 5:
            image = np.reshape(image, ((1,) + image.shape))
            # print(image.shape, '@@@@@@@@@')
        image = torch.from_numpy(image)
        image = image.to(device)
        # print(image.shape, '=' * 12)

    else:
        image = image.astype('float32')
        if numpy_flag:
            pass
        else:
            while len(image.shape) < dim:
                image = np.reshape(image, ((1,) + image.shape))
            image = torch.from_numpy(image)
            image = image.to(device)

    return image


def load_params(path, key_name='Params'):
    data = sio.loadmat(path)
    data = data[key_name]
    return data


def load_params_torch(path, key_name='Params',device='cpu', dim=5):
    """save the data on the specify dimension"""
    """ index list:   fov dim:0 voxSize:dim:1 sizeVol:dim:2 TAng:dim:345"""
    data = sio.loadmat(path)
    data = data[key_name]
    try:
        TAng = data[0,0]['TAng']  # the size of the TAng would 3*3
    except:
        print(path)
    fov = torch.tensor(data[0, 0]['fov'], dtype=torch.int64)
    voxSize = data[0, 0]['voxSize']  # the size of the voxSize would be 1*3
    sizeVol = data[0, 0]['sizeVol']  # the size of the voxSize would be 1*3
    sizeVol = sizeVol.astype(np.int64)
    TAng=torch.tensor(TAng, dtype=torch.float64)
    voxSize = torch.tensor(voxSize, dtype=torch.float64)
    sizeVol = torch.tensor(sizeVol, dtype=torch.int64)

    param_mat = torch.randn((6, 3))
    param_mat[0, :] = fov
    param_mat[1, :] = voxSize
    param_mat[2, :] = sizeVol
    param_mat[3:, :] = TAng
    if dim == 5:
        param_mat = torch.unsqueeze(param_mat, 0)
    param_mat = torch.unsqueeze(param_mat, 0)
    param_mat = param_mat.to(device)
    return param_mat


def get_default_params(data,dim=5):
    _,_,x,y,z = data.shape
    device = data.device
    param_mat = torch.randn((6, 3))
    param_mat[0, :] = torch.tensor([x,y,z])
    param_mat[1, :] =  torch.tensor([1, 1, 1])
    param_mat[2, :] =  torch.tensor([x,y,z])
    param_mat[3:, :] =  torch.tensor([[1,0,0], [0,1,0],[0,0,1]])
    if dim == 5:
        param_mat = torch.unsqueeze(param_mat, 0)
    param_mat = torch.unsqueeze(param_mat, 0)
    param_mat = param_mat.to(device)
    return param_mat


class MyDatasetPPM(torch.utils.data.Dataset):
    """ this dataset for return deltaB,Mag, label, param """
    def __init__(self, path_list, noise_flag=0,batch_size=1):
        self.path_list = path_list
        self.data_len = len(path_list)
        self.noise_flag = noise_flag
        self.batch_size = batch_size

    def __getitem__(self, index):
        try:
            deltaB = load_block_image(self.path_list[index], 'deltaB', dim=4)
            if self.noise_flag:
                deltaB=deltaB+1e-3*torch.randn([self.batch_size,64,64,32])
            mag = load_block_image(self.path_list[index], 'Mag', dim=4)
            label = load_block_image(self.path_list[index], 'chi_mo', dim=4)
            param = load_params_torch(self.path_list[index], 'Params', dim=4)
        except Exception as ep:
            print(self.path_list[index])
        return deltaB, mag, label, param

    def __len__(self):
        return self.data_len


def get_train_loader_PPM(path_list, sample_sum, the_batch_size, noise_flag=0):
    """  the data loader for Phase,Params,Mag,chi_mo  """
    path_list = get_path_list(path_list, sample_sum)
    total_num = len(path_list)
    # print_c(total_num)
    torch_dataset = MyDatasetPPM(path_list,noise_flag,the_batch_size)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,       # torch TensorDataset format
        batch_size=the_batch_size,   # mini batch size
        shuffle=True,                # random shuffle for training
        num_workers=8,               # sub processes for loading data
        pin_memory=True,
        drop_last=True
    )
    return train_loader, total_num


def get_path_from_txt(path):
    flag=0
    patt1_pound = '#.*'
    pattern1_pound = re.compile(patt1_pound)

    patt0_modelname = 'Model name:\s*(\w*)\s*'
    pattern0_model_name = re.compile(patt0_modelname)

    patt2_star = '\*\s*(\S*)\s*name:\s*(.*)\s*'
    pattern2_star = re.compile(patt2_star)

    patt2_star1 = '\*\s*(\S*)\s*'
    pattern2_star1 = re.compile(patt2_star1)

    patt3_train = '\+\+\+\+.*(Data Path for training).*\+\+\+\+'
    pattern3_train = re.compile(patt3_train)

    patt4_testGT = '\+\+\+\+.*(Test with ground truth).*\+\+\+\+'
    pattern4_testGT = re.compile(patt4_testGT)

    patt5_testnoGT = '\+\+\+\+.*(Test without ground truth).*\+\+\+\+'
    pattern5_testnoGT = re.compile(patt5_testnoGT)

    with open(path, 'r', encoding='utf-8') as f:
        """
        flag:  1:train  2:testGT   3:testnoGT     
        """
        train_path_list = []
        testGT_path_list = []
        testnoGT_path_list = []
        model_name=''
        while True:
            lineread = f.readline()
            result = pattern1_pound.findall(lineread)
            if result:
                ''' the comment line, skip'''
                continue
            result0=pattern0_model_name.findall(lineread)
            if result0:
                """get the model name """
                model_name=result0
            result1 = pattern3_train.findall(lineread)
            if result1:
                print(result1)
                flag = 1
            result2 = pattern4_testGT.findall(lineread)
            if result2:
                print(result2)
                flag = 2
            result3 = pattern5_testnoGT.findall(lineread)
            if result3:
                print(result3)
                flag = 3

            if flag == 1:
                result01 = pattern2_star1.findall(lineread)
                if result01:
                    print(result01)
                    train_path_list.append(result01)

            if flag == 2:
                result02 = pattern2_star.findall(lineread)
                if result02:
                    print(result02)
                    testGT_path_list.append(result02)

            if flag == 3:
                result03 = pattern2_star.findall(lineread)
                if result03:
                    print(result03)
                    testnoGT_path_list.append(result03)

            if not lineread:
                break
            pass
    return train_path_list,testGT_path_list,testnoGT_path_list,model_name[0]


def get_the_near_n(num,n_fold):
    res=num%n_fold
    return res


def data_reshape_for_test(data, param, n_fold, plus_clip=0, device='cpu'):
    """reshape the data and the params to the designed dimension"""
    """param is load from the load_params_torch """
    """plus_clip used when the data size is too large"""
    data_shape = data.shape
    res1 = data_shape[2] % n_fold+plus_clip*n_fold
    res2 = data_shape[3] % n_fold+plus_clip*n_fold
    res3 = data_shape[4] % n_fold

    dim1 = data_shape[2] - res1
    dim2 = data_shape[3] - res2
    dim3 = data_shape[4] + n_fold-res3 if res3 > 0 else data_shape[4]
    # print_c(param,color='purple')
    if param == None:
        pass
    else:
        param[0, 0, 2, :] = torch.as_tensor([dim1, dim2, dim3])
        param[0, 0, 0, :] = torch.mul(param[0, 0, 1, :], param[0, 0, 2, :])     # the slice can be add to dim3+7
        param = param.to(device)
    # param_mat[0, :] = fov
    # param_mat[1, :] = voxSize
    # param_mat[2, :] = sizeVol
    # param_mat[3:, :] = TAng
    # new_data = torch.zeros((1,1,dim1,dim2,dim3))
    # print('The shape of the data :', data.shape)
    # print('dim1,dim2,dim3',dim1,dim2,dim3)
    new_data = torch.zeros(1, 1, dim1, dim2, dim3)
    slice0 = slice((res1//2), -1*((res1//2+res1%2))) if res1 > 0 else slice(0, data_shape[2])
    slice1 = slice((res2//2), -1*(res2//2+res2%2))if res2 > 0 else slice(0, data_shape[3])
    slice2 = slice(0, -1*(n_fold-res3)) if res3>0 else slice(0, data_shape[4])
    new_data[:, :, :, :, slice2] = data[:, :, slice0, slice1, :]
    # resn = [res1, res2, res3]
    slice_num = [slice0, slice1, slice2]
    # print('new data shape', new_data.shape)
    new_data = new_data.to(device)
    return new_data, param, slice_num


def data_shape_recover(new_data, ori_data, slice_num):
    # (output, data_label, resn)
    """
    ori_data need to be 5D
    """
    # print('slice_num:',slice_num)
    data = torch.zeros_like(ori_data)
    slice0 = slice_num[0]
    slice1 = slice_num[1]
    slice2 = slice_num[2]
    data[0, 0, slice0, slice1, :] = new_data[0, 0, :, :, slice2]
    # print('                 data0.shape:', data.shape)
    return data


def get_path_list(main_path, sample_num=None, shuffle=False):
    data_list = glob.glob(main_path+'/*')  # get all data list
    print('The number of the total data pair is :', len(data_list))
    if len(data_list) == 0:
        print_c('Path:', main_path, 'contain nothing !!', color='purple')
    if sample_num==None:
        sample_num=len(data_list)
    if shuffle:
        random.shuffle(data_list)
    if sample_num == len(data_list):
        data_list_for_train = data_list
    elif sample_num > len(data_list):
        sample_num_x = sample_num
        sample_num = len(data_list)
        print_c(f'Warning: The sample_num:; {sample_num_x} should not larger than the total data pair'
                f' number: {len(data_list)},final sample num is: {len(data_list)} !', color='red')
        # sampled_data_list = random.sample(data_list, sample_num)
        data_list_for_train = data_list
    else:
        # get the data for training
        sampled_data_list = data_list[:sample_num]
        data_list_for_train = sampled_data_list

    print('data pair number for train is : ', len(data_list_for_train))
    return data_list_for_train


def fix_all_param(model, flag=0):
    for name, param in model.named_parameters():
        if flag:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model






