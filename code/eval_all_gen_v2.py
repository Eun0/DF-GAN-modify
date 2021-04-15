from __future__ import print_function


import os
import sys


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetG_V2
import torchvision.utils as vutils
from tqdm import tqdm
from collections import OrderedDict
from eval.fid_score import calculate_fid_given_paths

from miscc.utils import mkdir_p,truncated_z_sample,setup_logger
from miscc.config import cfg, cfg_from_file

from datasets import SBERTSentDataset

from DAMSM import RNN_ENCODER,CNN_ENCODER

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/tamsm_xent.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed',default=100)
    parser.add_argument('--num_samples',type=int,default=6000)
    parser.add_argument('--start_epoch',type=int,default=0)
    parser.add_argument('--metric',type=str,default='both')
    args = parser.parse_args()
    return args



def sampling(text_encoder, image_encoder,netG, dataloader,num_samples,metric,output_dir,logger):
    
 
    model_dir = f'{output_dir}/Models'
    
    model_list = sorted(glob.glob(f'{model_dir}/netG_*.pth'))[start_epoch:]

    results = {'r_epoch':0,'R_mean':0,'f_epoch':0,'fid':1000}
   
    for model in model_list:

        epoch = model.split('netG_')[-1].replace('.pth','')
        module_model_dict = torch.load(model,map_location='cpu')
        ddp = False
        for module in module_model_dict:
            if 'module.' in module:
                ddp = True
                break
            else:
                break

        if ddp:
            model_dict = OrderedDict()
            for key in module_model_dict: 
                k = key.split('module.')[-1]
                model_dict[k] = module_model_dict[key]
            netG.load_state_dict(model_dict)
        else:
            netG.load_state_dict(module_model_dict)

        netG.eval()
        netG.cuda()
        
        cnt = 0
        R_count = 0 
        R = np.zeros(num_samples)
        cont = True

        for ii in range(11):
            if (cont == False):
                break
            for data in tqdm(dataloader):
                if cont == False:
                    break 
                with torch.no_grad():
                    
                    imgs,sent_embs,keys = data
                    sent_embs = sent_embs.cuda()
                    
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    
                    #noise = torch.randn(sent_embs.size(0), 100)
                    #noise=noise.cuda()
                    noise = truncated_z_sample(sent_embs.size(0), 100,seed=100)
                    noise=torch.from_numpy(noise).float().cuda()
                    fake_imgs = netG(noise,sent_embs)

                    for j in range(sent_embs.size(0)):
                        im = fake_imgs[j].data.cpu().numpy()
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        
                        
                        fullpath = f'{img_dir}/{ii}_{keys[j]}.png'
                        im.save(fullpath)
                        cnt += 1

                    if cnt >= num_samples:
                        cont = False

                    s_r = ''
                    s_fid = ''

                    # if metric == 'r-precision' or metric == 'both':
                    #     _, cnn_code = image_encoder(fake_imgs)


                    #     for i in range(batch_size):
                    #         mis_captions,mis_captions_len = dataset.get_mis_caption(class_ids[i])
                    #         hidden = text_encoder.init_hidden(99)
                    #         _,sent_emb_t = text_encoder(mis_captions,mis_captions_len,hidden)
                    #         rnn_code = torch.cat((sent_embs[i, :].unsqueeze(0), sent_emb_t), 0)
                    #         scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                    #         cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                    #         rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                    #         norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                    #         scores0 = scores / norm.clamp(min=1e-8)
                    #         if torch.argmax(scores0) == 0:
                    #             R[R_count] = 1
                    #         R_count += 1
                        

                    #     if R_count >= num_samples:
                    #         sum = np.zeros(10)
                    #         np.random.shuffle(R)
                    #         assert num_samples%10 == 0

                    #         for i in range(10):
                    #             sum[i] = np.average(R[i * int(num_samples//10):(i + 1) * int(num_samples//10) - 1])
                    #         R_mean = np.average(sum)
                    #         R_std = np.std(sum)

                    #         s_r = f' R mean:{R_mean:.4f} std:{R_std:.4f} '
                            
                    #         if results['R_mean'] < R_mean:
                    #             results['r_epoch'] = epoch
                    #             results['R_mean'] = R_mean

                            
                    if metric == 'fid' or metric == 'both' and cnt>=num_samples:
                        paths=["",""]
                        paths[0] = f'eval/coco_val.npz'
                        paths[1] = f'{img_dir}/'
                        fid_value = calculate_fid_given_paths(paths, 50, True, 2048)
                        s_fid = f'FID: {fid_value}'
                        if fid_value < results['fid']:
                            results['f_epoch'] = epoch 
                            results['fid'] = fid_value

                    if cnt >= num_samples:

                        s = f'epoch : {epoch} {s_r} {s_fid}'
                        #print(s)
                        logger.info(s)

    s_res = f"Best models is {results['r_epoch']} with R mean : {results['R_mean']} {results['f_epoch']} with fid : {results['fid']}"
    logger.info(s_res)    
            


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    #print('Using config:')
    #pprint.pprint(cfg)

    output_dir = '../output/%s_%s' %\
        (cfg.DATASET_NAME,cfg.CONFIG_NAME)

    cfg.SEED = args.manualSeed

    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = False
    cudnn.deterministic = True

    eval_dir = output_dir + '/Eval'
    img_dir = eval_dir + '/imgs'
    log_dir = eval_dir + '/log'

    mkdir_p(output_dir)
    mkdir_p(img_dir)
    mkdir_p(log_dir)

    logger = setup_logger(cfg.CONFIG_NAME,log_dir)
    #logger = setup_logger(cfg.CONFIG_NAME,'')
    logger.info('Using config:')
    logger.info(cfg)

    # Get data loader ##################################################
    
    imsize = cfg.TREE.BASE_SIZE
    batch_size = 50
    image_transform = transforms.Compose([
            transforms.Resize((imsize,imsize)),
        ])
    print(f'Generate {imsize}x{imsize} images')
    print(f'Using {cfg.TEXT.ENCODER_NAME} text encoder')
    dataset = SBERTSentDataset(data_dir=cfg.DATA_DIR, mode='test',transform=image_transform,cfg=cfg)
    assert dataset
    print(f'dataset size : {len(dataset)}')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True,
        shuffle=False, num_workers=int(cfg.WORKERS))
    #dataloader = None

    # # validation data #

    netG = NetG_V2(cfg.TRAIN.NF, 100, cfg.TEXT.EMBEDDING_DIM).cuda()
    #netG = torch.nn.parallel.DistributedDataParallel(netG,device_ids=[cfg.GPU_ID],output_device=cfg.GPU_ID)
    
    text_encoder = None
    image_encoder = None


    #image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    #image_encoder.load_state_dict(torch.load(cfg.TEXT.ENCODER_NAME.replace('text','image'),map_location='cpu'))
    #image_encoder.cuda()
    # for p in image_encoder.parameters():
    #     p.requires_grad = False 
    # image_encoder.eval()

    start_epoch = args.start_epoch

    sampling(text_encoder,image_encoder, netG, dataloader,args.num_samples,args.metric,output_dir,logger)  # generate images for the whole valid dataset
        


        
