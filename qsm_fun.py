import torch
import torch.nn as nn
import numpy as np
from utils import fftshift_tensor,tensor_unsqueeze,ifftshift_tensor,get_mask_from_data
import torch.fft as tfft


def d_k_torch_batch(fov,mat_size,TAng, thre=0.2, device='cpu'):
    ''' get dipole kernel from fov,mat_size,TAng '''
    batch_size = mat_size.shape[0]
    # time_record1 = time.time()
    Nx = int(mat_size[0, 0, 0])
    Ny = int(mat_size[0, 0, 1])
    Nz = int(mat_size[0, 0, 2])
    dkx = 1 / fov[0, :, 0]
    dky = 1 / fov[0, :, 1]
    dkz = 1 / fov[0, :, 2]

    kx = torch.mul(dkx, torch.linspace(-(Nx / 2), Nx / 2 - 1, Nx, device=device))
    ky = torch.mul(dky, torch.linspace(-(Ny / 2), Ny / 2 - 1, Ny, device=device))
    kz = torch.mul(dkz, torch.linspace(-(Nz / 2), Nz / 2 - 1, Nz, device=device))
    [KY_Grid,KX_Grid,KZ_Grid] = torch.meshgrid(kx, ky, kz)

    KX_Grid = tensor_unsqueeze(KX_Grid, [0, 1])
    KY_Grid = tensor_unsqueeze(KY_Grid, [0, 1])
    KZ_Grid = tensor_unsqueeze(KZ_Grid, [0, 1])

    KSq = KX_Grid ** 2 + KY_Grid ** 2 + KZ_Grid ** 2
    # get the dipole one by one   Batch processing
    R31 = TAng[:, 0, 2, 0]
    R32 = TAng[:, 0, 2, 1]
    R33 = TAng[:, 0, 2, 2]
    # print_c(R31,R32,R33,color='blue')
    R31 = tensor_unsqueeze(R31, [1, 2, 3, 4])
    R32 = tensor_unsqueeze(R32, [1, 2, 3, 4])
    R33 = tensor_unsqueeze(R33, [1, 2, 3, 4])

    KZp_Grid = R31 * KX_Grid + R32 * KY_Grid + R33 * KZ_Grid
    C = 1 / 3 - ((KZp_Grid ** 2) / KSq)

    data_zero = torch.zeros([batch_size, 1, Nx, Ny, Nz], device=device)
    C = C.to(device)
    C = torch.where(torch.isnan(C), data_zero.float(), C.float())
    data_one = torch.ones([batch_size,1,Nx, Ny, Nz], device=device)
    G_mask = torch.where((torch.abs(C) < thre), data_one.float(), data_zero.float())   # 1 for cone
    return C, G_mask


def get_dk_torch_batch(params, device='cpu', train_flag=0, s=(224, 224, 126)):
    """  fov dim:0 voxSize:dim:1 sizeVol:dim:2 TAng:dim:345"""
    voxSize = params[0, :, 1, :][0]
    # print('voxSize',voxSize)
    if train_flag:
        params[:, :, 2, :] = torch.tensor([s[0], s[1], s[2]], device=device)
        # params[:, :, 2, :] = torch.tensor([205, 164, 205], device=device)
        mat_size = params[:, :, 2, :]
    else:
        mat_size = params[:, :, 2, :]

    if train_flag:
        params[:, :, 0, :] = torch.tensor([voxSize[0]*s[0],voxSize[1]*s[1], voxSize[2]*s[2]], device=device)
        # params[:, :, 0, :] = torch.tensor([205, 164, 205], device=device)
        fov = params[:, :, 0, :]
        # print(fov.shape)  # [x ,1 ,3]
    else:
        fov = params[:, :, 0, :]

    TAng = params[:, :, 3:, :]

    # get dipole_k,G_mask
    dipole_k, G_mask = d_k_torch_batch(fov, mat_size, TAng, 0.05, device)
    # print('dipole_k.shape',dipole_k.shape)
    return dipole_k, G_mask


def get_tkd(deltab, thresh, params, train_flag=1, s=(224, 224, 126)):
    ''' get tkd result from deltab'''
    device = deltab.device
    data_shape = deltab.shape
    if train_flag:
        fft_data = tfft.fftn(deltab, dim=[2, 3, 4], s=s)  # fft as the shape
    else:
        # use the input data.shape  recalculate the params
        fft_data = tfft.fftn(deltab, dim=[2, 3, 4])
        params[:, :, 2, :] = torch.tensor([data_shape[2], data_shape[3], data_shape[4]], device=device)  # matrix
        params[:, :, 0, :] = params[:, :, 2, :] * params[:, :, 1, :]   # fov
    dipole_k, G_mask = get_dk_torch_batch(params, device=device, train_flag=train_flag,s=s)
    """ calculate the tkd  from the deltab """
    delta_k = fftshift_tensor(fft_data, device)
    delta_k = delta_k.to(device)
    dk = dipole_k.to(device)
    dk_inv = torch.where((torch.abs(dk) > thresh), torch.div(1, dk), torch.zeros_like(dk))
    ''' in the cone the data would be zero '''
    chi_k = delta_k * dk_inv
    chi_k_s = ifftshift_tensor(chi_k, device=device)
    chi_s = tfft.ifftn(chi_k_s, dim=[2, 3, 4])
    chi_s = chi_s.real
    if train_flag:
        chi_result = chi_s[:, :, 0:data_shape[2], 0:data_shape[3], 0:data_shape[4]]
    else:
        chi_result = chi_s
    mask = get_mask_from_data(deltab)
    chi_result = chi_result * mask
    return chi_result













