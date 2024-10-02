import argparse
import logging
import math
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import core.logger as Logger
import data as Data
import model as Model
import core.metrics as Metrics
from decimal import Decimal
import pandas as pd

"""
python time_train.py \
-c config/shinwa_time_train.json \
-p train \

"""

# train model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/smap_time_train.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)

    test_set = Data.create_dataset(opt['datasets']['test'], 'test')

    test_loader = Data.create_dataloader(test_set, opt['datasets']['test'], 'test')
    logger.info('Initial Dataset Finished')

    logger.info('Initial Dataset Finished')

    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    save_model_iter = math.ceil(train_set.__len__() / opt['datasets']['train']['batch_size'])
    print('save_model_iter:', save_model_iter)
    model_epoch = 100
    logger_name = 'test' + str(model_epoch)
    logger_test = logging.getLogger(logger_name)
    params = {
        'opt': opt,
        'logger': logger,
        'logger_test': logger_test,
        'model_epoch': model_epoch,
        'row_num': test_set.row_num,
        'col_num': test_set.col_num
    }
    start_label = opt['model']['beta_schedule']['test']['start_label']
    end_label = opt['model']['beta_schedule']['test']['end_label']
    step_label = opt['model']['beta_schedule']['test']['step_label']
    step_t = opt['model']['beta_schedule']['test']['step_t']
    strategy_params = {
        'start_label': start_label,
        'end_label': end_label,
        'step_label': step_label,
        'step_t': step_t
    }
    all_datas = pd.DataFrame()
    sr_datas = pd.DataFrame()
    differ_datas = pd.DataFrame()
    while current_epoch < n_epoch:
        current_epoch += 1
        for _, train_data in enumerate(tqdm(train_loader)):
            current_step += 1
            if current_epoch > n_epoch:
                break
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if current_epoch % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # save model
            idx = 0
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0 and current_step % save_model_iter == 0:
                for _, test_data in enumerate(test_loader):
                    idx += 1
                    diffusion.feed_data(test_data)
                    diffusion.test(continous=False)
                    visuals = diffusion.get_current_visuals()

                    all_data, sr_df, differ_df = Metrics.tensor2allcsv(visuals, params['col_num'])
                    all_datas = Metrics.merge_all_csv(all_datas, all_data)
                    sr_datas = Metrics.merge_all_csv(sr_datas, sr_df)
                    differ_datas = Metrics.merge_all_csv(differ_datas, differ_df)
                    print("idx", idx)

                all_datas = all_datas.reset_index(drop=True)
                sr_datas = sr_datas.reset_index(drop=True)
                differ_datas = differ_datas.reset_index(drop=True)


                for i in range(params['row_num'], all_datas.shape[0]):
                    all_datas.drop(index=[i], inplace=True)
                    sr_datas.drop(index=[i], inplace=True)
                    differ_datas.drop(index=[i], inplace=True)

                f1,accuracy,precision, recall  = Metrics.relabeling_strategy(all_datas, strategy_params)
                # 正解率を出力
                print('Accuracy: {:.4f}'.format(accuracy))

                temp_f1 = Decimal(f1).quantize(Decimal("0.0000"))

                print('F1-score: ', float(temp_f1))
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)

    logger.info('End of training.')
