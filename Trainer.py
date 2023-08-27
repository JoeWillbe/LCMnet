from data_fun_torch import *
from utils import *
import random
import numpy as np
import datetime
import time
import traceback
from tensorboardX import SummaryWriter
torch.autograd.set_detect_anomaly(True)    # Derivative anomaly detection
# logging setting


class Trainer():
    def __init__(self, arg_set, model):
        self.load_flag = arg_set.get('load_flag')
        self.version = arg_set.get('version', 'default')
        self.data_set = arg_set.get('dataset', 'D07')
        self.weight_load = arg_set.get('weight_load', 'default_weight_load')
        self.weight_save = arg_set.get('weight_save', 'default_weight_save')
        self.save_flag = arg_set.get('save_flag', 1)
        self.data_pair_num = arg_set.get('data_num', 18000)
        self.device_list = arg_set.get('device_list')    # get a list of device
        self.device = self.device_list[0]                # the default
        # training setting
        self.num_epoch = arg_set.get('num_epoch', 30)
        self.batch_size = arg_set.get('batch_size', 4)
        self.learning_rate = arg_set.get('lr', 0.0001)      # 1e-4
        self.epoch_val_step = 5  # the interval between the validation
        self.model = model
        self.project_path = '/home/joe/File/python_file/PC_sync/pytorch_1_0/DL_QSM_1'
        self.main_path = arg_set.get('main_path', '/home/joe/File/DATA/')   
        self.loss_fun = LossHFEN_MSE(self.device)
        self.log_path = arg_set.get('log_path')
        self.test_info_path = '/home/joe/File/python_file/PC_sync/pytorch_1_0/DL_QSM_1/Analysis/test_info_log.txt'
        print_c(' ----------------- init -----------------')
        print('self.save_flag', self.save_flag, 'self.load_flag', self.load_flag)
        print('self.learning_rate', self.learning_rate, 'self.num_epoch', self.num_epoch)
        print('self.batch_size', self.batch_size, 'self.device', self.device)
        print('self.log_path', self.log_path, 'self.model', type(self.model))
        print_c(' ----------------- init -----------------')

    def load_weight(self, load_flag):
        if load_flag:
           self.model.load_state_dict(torch.load('model_save/'+self.weight_load+'.pkl', map_location=torch.device('cpu')))
           print(('Load weight from: %s' % ('model_save/'+self.weight_load+'.pkl')))

    def save_weight(self, save_flag):
        if save_flag:
            torch.save(self.model.state_dict(), 'model_save/'+self.weight_save+'.pkl')
            print(' ** ---------------- model save completed  ----------------- **')

    def get_param_num(self):
        ''' calculate the numeber of params'''
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.model.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def save_model(self, model_name):
        now = datetime.datetime.now()
        date = '_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day)
        path = './model_whole/' + model_name+date+'.pkl'
        self.load_weight(1)
        torch.save(self.model, path)

    def load_model(self, model_path):
        self.model = torch.load(model_path)
        return self.model

    def save_log_file(self, mode, str1=None, str2=None, loss_fun=None, flag=0):
        """ Save the log file mainly for training """
        now = datetime.datetime.now()
        time_str1 = ('%d-%02d-%02d %02d:%02d:%02d' % (now.year, now.month, now.day, now.hour, now.minute, now.second))
        print('Save the log file ', time_str1)
        if flag == 1:
            # flag == 1 for start
            now = datetime.datetime.now()
            time_record = '{}/{}/{} {}:{}:{} '.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
            logging.info('***'*20)
            logging.info('======================       Start     ======================')
            logging.info('* Start at:    %s' % time_record)
            logging.info('* Model name:  %s' % str(type(self.model)))
            str01 = 'Num of Epoch: %d' % self.num_epoch + '  Learning rate: %.4e' % self.learning_rate + ' Batch size: %d ' % self.batch_size
            str02 = 'Training data path:' + str(self.path_train)
            str03 = 'Device :' + str(self.device)
            logging.info(str01)
            logging.info(str02)
            logging.info(str03)
            logging.info('* The mode is: %s' % mode)
            if loss_fun:
                logging.info('* The loss function is :    %s' % str(type(loss_fun)))
            str_load = '* The weight file for loading is : %s ' % str(self.weight_load) + 'Load flag:', str(self.load_flag)
            str_save = '* The weight file for saving is :  %s ' % str(self.weight_save) + 'Save flag:', str(self.save_flag)
            logging.info(str_load)
            logging.info(str_save)

        if flag == 2:
            # flag == 2 for end
            now = datetime.datetime.now()
            time_record = '{}/{}/{},{}:{}:{} '.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
            logging.info('Ending at {}'.format(time_record))
            logging.info('====='*100)

        if mode == 'train':
            if str1:
                # print the str1 when training
                logging.info(str1)

        elif mode == 'test':
            if str2:
                # print the str2 when testing
                logging.info(str2)

    def get_path(self):
        ''' get the test path and the train path '''
        self.path_train = self.main_path + 'Train/'
        self.path_val = self.main_path + 'Validation/'     # for the full size data
        print_c('* self.path_train :', self.path_train, color='purple')

    def init_optimizers(self, weight_decay=5e-5):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)

    def logging_config(self, flag=1):
        now = datetime.datetime.now()
        if flag:
            log_path = './log_file/{}_{}_{}_{}.log'.format(now.year, now.month, now.day,now.hour)
        else:
            log_path = './log_file/QSM_{}_{}.log'.format(self.version,self.data_set)
        if self.log_path:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(levelname)s %(message)s',
                                # format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                                datefmt='%a %d %b %Y %H:%M:%S',
                                filename=self.log_path,
                                filemode='a'
                                )
        else:
            print('using the date.log')
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s %(levelname)s %(message)s',
                                # format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                                datefmt='%a %d %b %Y %H:%M:%S',
                                filename=log_path,
                                filemode='a'
                                )

    def mat_save(self, dict, dataname):
        save_path = './Training_record/'+ dataname +'.mat'
        sio.savemat(save_path, dict)

    def init_weight(self):
        for name, params in self.model.named_parameters():
            if 'bias' in name:
                torch.nn.init.normal_(params, 0, 0.001)

    def test_with_GT(self, save_result=1, Mag_flag=0, mask_flag=0):
        """  default test : with ground truth """
        result_path = '/home/joe/File/python_file/DL_QSM/result/'
        # ------------------  qsm 2016  --------------
        test_path = '/home/joe/File/DATA/data/Test_with_GT_SIM/qsm2016_chi33.mat'
        save_path = result_path + self.version + '_qsm2016_chi33.mat'
        print('save_path: ', result_path)
        #  test, calculate metric and save result
        if Mag_flag:
            ''' for model with mag data as input'''
            test_loss = self.test_with_labelPPM(test_path, save_result, save_path, 1, 1, 16)
        else:
            ''' for model w/o mag data as input'''
            test_loss = self.test_with_label(test_path, save_result, save_path, mask_flag, 1, 16)
        print('Test completed !')

    def test_without_GT(self, data_list, save_result=1, mag_flag=0,mask_flag=0):
        """test on the data without ground truth"""
        self.get_path()
        self.logging_config()
        self.model.eval()
        model_version = self.version
        result_path0 = '/home/joe/File/python_file/PC_sync/pytorch/DL_QSM_1/Result/'
        the_time = time.strftime("%Y%m%d", time.localtime())
        for item in data_list:
            test_data_path = '/home/joe/File/DATA/data/Test_without_GT/'+item
            result_path = result_path0 + model_version+'{}_'.format(the_time)+item
            print('result path ', result_path)
            self.test_without_label(test_data_path, save_result, result_path,mask_flag,n_fold=16,mag_flag=mag_flag)

    def train_ppm(self, print_flag=2, sample_num=None, epoch_set=None):
        """ Training:  for the training data with Phase Params Mag  """
        self.get_path()
        self.logging_config()
        self.init_optimizers()
        self.init_weight()
        if print_flag > 0:
            self.save_log_file('Train', loss_fun=self.loss_fun, flag=1)
        writer = SummaryWriter('board_log/Train' + self.data_set + '_' + self.version + '_1_3')
        if sample_num == None:
            sample_num = self.data_pair_num
        else:
            sample_num = sample_num
        if epoch_set == None:
            epoch_set = self.num_epoch
        else:
            epoch_set = epoch_set
        train_loader, total_num = get_train_loader_PPM(self.path_train, sample_num, self.batch_size)
        val_num = 426   # 55
        val_loader, val_data_num = get_train_loader_PPM(self.path_val, val_num, self.batch_size)
        if print_flag > 0:
            self.save_log_file('Train', loss_fun=self.loss_fun, flag=1)
        try:
            self.load_weight(self.load_flag)
        except Exception as err:
            print('Error when loading weights !', err)
            str_load = 'Error when loading weights ! ' + str(err)
            self.save_log_file('Train', str_load)
        self.model.to(self.device)
        print('Start training ......')
        mean_loss = 10
        val_loss = 10
        global_step = 0
        start_flag = 0
        loss_temp = 10
        loss_temp_v = 10
        epoch_bias = 0
        global_step_list = []
        mean_loss_list = []
        val_loss_list = []
        epoch_list = []

        for epoch in range(1, 1+epoch_set+epoch_bias):
            if epoch > 20:
                for p in self.optimizer.param_groups:
                    p['lr'] = 1e-5
            loss_sum = 0
            for step, (x_deltab , x_mag, y, param) in enumerate(train_loader):
                # deltaB, mag, label, param
                data_trained = min(step * self.batch_size, total_num)
                x_deltab = x_deltab.to(self.device)
                x_mag = x_mag.to(self.device)
                y = y.to(self.device)
                param = param.to(self.device)
                out = self.model(x_deltab, x_mag, param, 1)
                loss = self.loss_fun(out, y)
                # loss = self.loss_fun2(out, y, param, 1, s=(224,224,126))     # loss_forward_mse
                loss_sum = loss_sum + loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step_n = 40
                if step % step_n == 0 and step >0:
                    v_loss = 0
                    count_val = 0
                    with torch.no_grad():
                        for step1, (x_delatb1, x_mag1, y1, param1) in enumerate(val_loader):
                            y1 = y1.to(self.device)
                            x_mag1 = x_mag1.to(self.device)
                            x_delatb1 = x_delatb1.to(self.device)
                            param1 = param1.to(self.device)
                            out1 = self.model(x_delatb1, x_mag1, param1, 1)
                            loss1 = self.loss_fun(out1, y1, print_flag=0)
                            v_loss = v_loss + loss1.item()
                            count_val = count_val + 1
                    val_loss = v_loss / count_val
                    print_c(' Val_loss:%.4e' % val_loss, color='blues')
                    global_step = global_step + 1
                    mean_loss = 1 / step_n * loss_sum
                    loss_sum = 0
                    print('global_step:', global_step)
                    # test on the val block to get metric
                    writer.add_scalar('Train_loss', mean_loss, global_step=global_step)
                    writer.add_scalar('Validation_loss', val_loss, global_step=global_step)
                    writer.add_scalar('Learning_rate', self.learning_rate, global_step=global_step)
                    str_v = 'Validation loss: %.4e' % val_loss
                    self.save_log_file('Train', str_v)
                    # save data
                    epoch_list.append(epoch)
                    global_step_list.append(float(global_step))
                    mean_loss_list.append(mean_loss)
                    val_loss_list.append(val_loss)
                    dict_0 = {'global_step_list': global_step_list, 'mean_loss_list': mean_loss_list,
                              'val_loss_list': val_loss_list, 'epoch_list': epoch_list}
                    now_x = datetime.datetime.now()
                    dataname = self.version + '_{}_{}'.format(now_x.year, now_x.month)
                    self.mat_save(dict_0, dataname)

                if step % 20 == 0 and print_flag > 0:
                    'Print log at the ending of every 20 steps'
                    str01 = '*Loss: %.4e' % loss + '  Learning rate:' + str(self.learning_rate)
                    str02 = '  Data number: {}/ {}'.format(data_trained, total_num)
                    str1 = str01 + str02
                    self.save_log_file('train', str1)

                if epoch == 0 and start_flag == 1:
                    loss_temp = mean_loss
                    start_flag = 0

                if mean_loss < loss_temp:
                    loss_temp = mean_loss
                    print_c('Epoch: %d ' % (epoch), '  Mean loss: %.3e' % mean_loss, color='yellows')
                    print_c('Save model ...')
                    now = datetime.datetime.now()
                    torch.save(self.model.state_dict(), './model_save/QSM_S_' +self.weight_save + "best_Val" + '_{}_{}.pkl'.format(now.month, now.day))


                if val_loss < loss_temp_v:
                    loss_temp_v = val_loss
                    print_c('Epoch: %d ' % (epoch ), '  val_loss: %.3e' % val_loss, color='yellows')
                    print_c('Save model ... ')
                    torch.save(self.model.state_dict(), self.weight_save + "best_Val" + '.pkl')

                if step % 50 == 0 and step > 1:
                    print_c('Epoch: ', epoch, '  save weights')
                    self.save_weight(self.save_flag)

            if epoch % 10 == 0 and epoch > 1 or epoch == 5 or epoch == 25:
                torch.save(self.model.state_dict(), self.weight_save + "Epoch_" + str(epoch) + '.pkl')
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param, epoch + epoch_bias)
            str_e = 'Epoch:' + str(epoch + epoch_bias)
            self.save_log_file('Train', str_e)
            self.save_weight(self.save_flag)
        writer.close()
        if print_flag > 0:
            self.save_log_file('train', flag=2)
        now = datetime.datetime.now()
        print_c('save weight')
        self.save_weight(self.save_flag)
        print('The ending time is {}-{}-{} {}:{}'.format(now.year, now.month, now.day, now.hour, now.minute))

    def test(self, path_txt=None, save_result=1, flag_mode=1,Mag_flag=1, mask_flag=1):
        '''
           batch test
        '''
        result_path0 = self.project_path+'/Result/'
        if path_txt:
            _, testGT_path_list, testnoGT_path_list, model_name = get_path_from_txt(path_txt)
            if testGT_path_list and flag_mode > 0:
                print('The number of the data to be test with GT is', len(testGT_path_list))
                for i in range(len(testGT_path_list)):
                    test_path = testGT_path_list[i][0][0]
                    result_name = testGT_path_list[i][0][1]
                    try:
                        """  default test : with ground truth  """
                        # self.logging_config()
                        the_time = time.strftime("%Y%m%d", time.localtime())
                        result_path = result_path0 + self.version + '_' + result_name + '_{}.mat'.format(the_time)
                        print('===========================')
                        print_c('*test path:', test_path, '\n*result name:', result_name, '\n*result path:',
                                result_path)
                        if Mag_flag:
                            '''for model input with mag data'''
                            self.test_with_labelPPM(test_path, save_result, result_path, 0, 1, 16)
                        else:
                            '''for model input w/o mag data'''
                            self.test_with_label(test_path, save_result, result_path, mask_flag, 1, 16)
                        print('Test with GT completed !')
                    except Exception as ep:
                        eprint_detail = traceback.format_exc()
                        print_c(eprint_detail)
                        print_c('Error when test the ', test_path, '\n', ep, color='red')

            if testnoGT_path_list and flag_mode > 0:
                print('The number of the data to be test without GT is', len(testnoGT_path_list))
                """ test on the data without label"""
                for i in range(len(testnoGT_path_list)):
                    test_path = testnoGT_path_list[i][0][0]
                    result_name = testnoGT_path_list[i][0][1]
                    try:
                        the_time = time.strftime("%Y%m%d", time.localtime())
                        result_path = result_path0 + self.version + '_' + result_name + '_{}.mat'.format(the_time)
                        print('*test path:', test_path, '*result name:', result_name, '*result path:', result_path)
                        if Mag_flag:
                            self.test_without_label(test_path, save_result, result_path, 0, 1, 16)
                        else:
                            self.test_without_label(test_path, save_result, result_path, 0, 1, 16)
                        print('Test without GT completed !')
                    except Exception as ep:
                        eprint_detail = traceback.format_exc()
                        print_c(eprint_detail)
                        print_c('Error when test the ', test_path, ep, color='red')
        else:
            print('no path_txt were found !')

    def test_with_label(self, path, save_result, result_path, mask_flag=0, write_excel_flag=0, n_fold=8):
        ''' test with label param_flag = 1 '''
        self.logging_config()
        now = datetime.datetime.now()
        self.model = self.model.to(self.device)
        # self.weight_path_load
        # load deltaB chi_mo Mask
        try:
            self.load_weight(self.load_flag)
        except Exception as err:
            print_c('* Error when loading weights', err, color='reds')
        self.model.eval()
        start_time_load = time.time()
        try:
            data_input = load_block_image(path, 'deltaB_o')
        except:
            data_input = load_block_image(path, 'deltaB')
        try:
            data_label = load_block_image(path, 'chi_mo')
        except:
            data_label = load_block_image(path, 'label')
        try:
            data_params = load_params_torch(path, 'Params')
            data_params = data_params.to(self.device)
        except:
            print_c('No params loading. Use the default params')
            data_params = get_default_params(data_input)
            data_params = data_params.to(self.device)
        try:
            data_mask = load_block_image(path, 'mask')
        except:
            data_mask = load_block_image(path, 'maskErode')
        finally:
            print_c('Loading mask error ！', color='cyans')
        data_mask = data_mask.to(self.device)
        data_label = data_label.to(self.device)
        data_input = data_input.to(self.device)
        default_type = torch.float32
        # Test setting
        plus_clip = 2
        time_load=time.time()
        print('time_load time:', time_load - start_time_load)
        time_x=time.time()
        input_reshape, params, resn = data_reshape_for_test(data_input, data_params, n_fold, plus_clip,
                                                            device=self.device)
        mask_reshape, params, resn = data_reshape_for_test(data_mask, data_params, n_fold, plus_clip,
                                                            device=self.device)
        time_y=time.time()
        print('shear time:', time_y-time_x)
        print_c('The input after sheared:', input_reshape.shape, resn)
        self.model = self.model.type(default_type)
        data_input = input_reshape.to(self.device)

        with torch.no_grad():
            self.model = self.model.to(self.device)
            print('Test 3D full size')
            start_time0 = time.time()
            if mask_flag:
                output = self.model(data_input, params, mask_reshape,1)
            else:
                output = self.model(data_input, params, 1)
            end_time0 = time.time()
            time_cost = end_time0 - start_time0
            end_time_out = time.time()
            print('The time cost is : ', time_cost)
            print('The full time cost is : ', end_time_out-start_time_load)
            data_inv_shear = data_shape_recover(output, data_label, resn)
            # print(data_inv_shear.shape)
            output = data_inv_shear*data_mask
            # print(data_inv_shear.shape, data_label.shape)
            # label_sheared, params, resn = data_reshape_for_test(data_input, data_params, n_fold)
            test_loss = self.loss_fun(output, data_label, self.test_info_path, self.weight_load)
            print_c('The test loss for full size data is :', test_loss, color='green')
            psnr, rmse, ssim, hfen = cal_4metrics(output, data_label)
            print_c('psnr: %.4f ' % psnr, 'rmse: %.4f ' % rmse, 'ssim: %.4f ' % ssim, 'hfen: %.4f ' % hfen,color='green')

            if save_result:
                # out0_save=(out0.cpu()).dectach().numpy()  # ,'out0_save': out0_save
                deltab_save = np.squeeze((data_input.cpu()).detach().numpy())
                pred_y_save = ((output.cpu()).detach().numpy())
                label_y_save = ((data_label.cpu()).detach().numpy())
                pred_y_mat = np.squeeze(pred_y_save)
                cos_v_mat = np.squeeze(label_y_save)
                y_dict = {'x': pred_y_mat, 'label': cos_v_mat, 'deltab': deltab_save}
                sio.savemat(result_path, y_dict)
                print(' ==================   Data model_save completed !   ============')
        return test_loss

    def test_with_labelPPM(self, path, save_result, result_path, param_flag=0, write_excel_flag=0, n_fold=8):
        ''' test with Phase Params Mag (PPM)'''
        self.logging_config()
        self.model.eval()
        data_input = load_block_image(path, 'deltaB')
        try:
            data_label = load_block_image(path, 'chi_mo')
        except:
            data_label = load_block_image(path, 'label')
        try:
            data_params = load_params_torch(path, 'Params')
        except:
            print_c('No params loading. Use the default params')
            data_params = get_default_params(data_input)
            data_params = data_params.to(self.device)
        try:
            data_mask = load_block_image(path, 'mask')
        except:
            data_mask = load_block_image(path, 'maskErode')
        finally:
            print_c('Loading mask error ！', color='cyans')
        try:
            data_mag = load_block_image(path, 'Mag')
        except:
            data_mag = load_block_image(path, 'magmap')
            print('Load magmap')
        # deltaB Mag Params flag
        data_mask = data_mask.to(self.device)
        data_mask0 = data_mask
        data_params = data_params.to(self.device)
        data_label = data_label.to(self.device)
        data_mag = data_mag.to(self.device)
        default_type = torch.float32
        # Test setting
        plus_clip = 1
        input_reshape, params, resn = data_reshape_for_test(data_input, data_params, n_fold, plus_clip,
                                                            device=self.device)
        data_mag, _, _ = data_reshape_for_test(data_mag, data_params, n_fold, plus_clip,
                                                            device=self.device)
        data_mask, _, _ = data_reshape_for_test(data_mask, data_params, n_fold, plus_clip,
                                                            device=self.device)
        print_c('The input after sheared:', input_reshape.shape, resn)
        self.model = self.model.type(default_type)
        # load weight
        try:
            self.load_weight(self.load_flag)
        except Exception as err:
            print_c('* Error when loading weights', err, color='reds')
        slice_num = input_reshape.shape[4]
        slice_set = 256   # limit the slice number of the inputs

        input_reshape = input_reshape.to(self.device)
        if slice_num <= slice_set:
            split_flag = 0
            data_input1 = input_reshape[:, :, :, :, :]
        else:
            split_flag = 1
            data_input1 = input_reshape[:, :, :, :, 0:slice_set]

        with torch.no_grad():
            self.model = self.model.to(self.device)
            if split_flag:
                print('Do nothing')
            else:
                print('Test 3D full size')
                start_time0 = time.time()
                # output = self.model(data_input1, params, data_mask)
                output = self.model(data_input1, data_mag, params=params,flag=1)
                end_time0 = time.time()
                time_cost = end_time0 - start_time0
            print('The time cost is : ', time_cost)
            data_inv_shear = data_shape_recover(output, data_label, resn)
            output = data_inv_shear*data_mask0
            test_loss = self.loss_fun(output, data_label, self.test_info_path, self.weight_load)
            print_c('The test loss for full size data is :', test_loss, color='green')
            psnr, rmse, ssim, hfen = cal_4metrics(output, data_label)
            print_c('psnr: %.4f ' % psnr, 'rmse: %.4f ' % rmse, 'ssim: %.4f ' % ssim, 'hfen: %.4f ' % hfen,color='green')
            if save_result:
                deltab_save = np.squeeze((data_input.cpu()).detach().numpy())
                pred_y_save = ((output.cpu()).detach().numpy())
                label_y_save = ((data_label.cpu()).detach().numpy())
                pred_y_mat = np.squeeze(pred_y_save)
                cos_v_mat = np.squeeze(label_y_save)
                y_dict = {'x': pred_y_mat, 'label': cos_v_mat,'deltab':deltab_save}
                sio.savemat(result_path, y_dict)
                print(' ==================   completed !   ============')
        return test_loss

    def test_without_label(self, path, save_result, result_path, mask_flag=0, n_fold=16,mag_flag=0):
        ''' test_with_label for the  '''
        self.logging_config()
        now = datetime.datetime.now()
        date = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + ' ' + str(now.hour) + ':' + str(now.minute)
        model_name = str(type(self.model))
        loss_type = str(type(self.loss_fun))
        print(model_name)
        self.model = self.model.to(self.device)
        try:
            data_input = load_block_image(path, 'deltaB_o')
        except:
            data_input = load_block_image(path, 'deltaB')
        try:
            mag = load_block_image(path, 'Mag')
        except:
            mag = load_block_image(path, 'magmap')
        try:
            data_mask = load_block_image(path, 'mask')
            data_mask = data_mask.to(self.device)
        except:
            data_mask = load_block_image(path, 'maskErode')
            data_mask = data_mask.to(self.device)
        finally:
            data_mask = get_mask_from_data(data_input)
            data_mask = data_mask.to(self.device)
            print_c('Get mask from input ！', color='cyans')
        try:
            data_params = load_params_torch(path, 'Params')
        except:
            print_c('No params loading.')
            data_params = torch.randn((1, 1, 6, 3))
        data_params = data_params.to(self.device)
        default_type = torch.float32
        # Test setting
        plus_clip = 2
        input_reshape, params, resn = data_reshape_for_test(data_input, data_params, n_fold, plus_clip,
                                                            device=self.device)
        mag_reshape, params, resn = data_reshape_for_test(mag, data_params, n_fold, plus_clip,
                                                            device=self.device)
        mask_reshape, params, resn = data_reshape_for_test(data_mask, data_params, n_fold, plus_clip,
                                                          device=self.device)
        print_c('The input after sheared:', input_reshape.shape, resn)
        self.model = self.model.type(default_type)
        # load weight
        try:
            self.load_weight(self.load_flag)
        except Exception as err:
            print_c('* Error when loading weights', err, color='reds')

        input_reshape = input_reshape.to(self.device)

        with torch.no_grad():
            start_time0 = time.time()
            if mag_flag:
                input_shape = input_reshape.shape
                print('Mag was used', input_shape)
                output = self.model(input_reshape,mag_reshape, params, 1, s=(input_shape[0], input_shape[1], input_shape[2]))
            else:
                if mask_flag:
                    output = self.model(input_reshape, params,mask_reshape, 1)
                else:
                    output = self.model(input_reshape, params, 1)
            end_time0 = time.time()
            time_cost = end_time0 - start_time0
            print('The time cost is : ', time_cost)

            data_inv_shear = data_shape_recover(output, data_input, resn)
            data_mask = data_shape_recover(mask_reshape, data_input, resn)
            output = data_inv_shear*data_mask

            if save_result:
                # out0_save=(out0.cpu()).dectach().numpy()  # ,'out0_save': out0_save
                deltab_save = np.squeeze((data_input.cpu()).detach().numpy())
                pred_y_mat = np.squeeze((output.cpu()).detach().numpy())
                y_dict = {'x': pred_y_mat, 'deltab': deltab_save}
                sio.savemat(result_path, y_dict)
                print(' ==================   Data model_save completed !   ============')
        return 0









