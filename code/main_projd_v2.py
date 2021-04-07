from __future__ import print_function

from miscc.utils import mkdir_p,setup_logger
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from DAMSM import RNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz

import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetProjD_V2
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--resume_epoch',type=int,default=0)
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader,device):
    
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    netG.load_state_dict(torch.load('models/%s/netG.pth'%(cfg.CONFIG_NAME)))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise=noise.to(device)
                fake_imgs = netG(noise,sent_emb)
            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp,i)
                im.save(fullpath)



def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device,output_dir,logger):
    #i = -1
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
         
        for step, data in enumerate(dataloader, 0):
            #i+=1
            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            imgs=imags[0].to(device)
            real_features = netD(imgs)
            output = netD.COND_DNET(real_features,sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()
            
            # synthesize fake images
            noise = torch.randn(batch_size, 100)
            noise=noise.to(device)
            fake = netG(noise,sent_emb)

            #if not cfg.TRAIN.ONLY_REAL:
            # G does not need update with D

            #if cfg.TRAIN.ONLY_REAL:
            #    for p in netD.COND_DNET.parameters():
            #        p.requires_grad_(False)

            fake_features = netD(fake.detach()) 
            errD_fake = netD.COND_DNET(fake_features,sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            #if cfg.TRAIN.ONLY_REAL:
            #    for p in netD.COND_DNET.parameters():
            #        p.requires_grad_(True)          

            errD = errD_real + (errD_fake + errD_mismatch)/2.0
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            #MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.COND_DNET(features,sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated,sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0,grad1),dim=1)                        
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()
            
            # update G

            #if (i+1) % cfg.TRAIN.N_CRITIC == 0:
                #i = -1
            features = netD(fake)
            output = netD.COND_DNET(features,sent_emb)
            errG = - output.mean()
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            errG.backward()
            optimizerG.step()

            logger.info('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f errD_real %.3f errD_mis %.3f errD_fake %.3f magp %.3f'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item(),errD_real.item(),errD_mismatch.item(),errD_fake.item(),d_loss_gp.item()))

        vutils.save_image(fake.data,
                        '%s/imgs/fake_samples_epoch_%03d.png' % (output_dir, epoch),
                        normalize=True)

        if epoch%1==0:
            torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (output_dir, epoch))
            torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (output_dir, epoch))
            torch.save(optimizerG.state_dict(),'%s/models/optimizerG.pth' %(output_dir))
            torch.save(optimizerD.state_dict(),'%s/models/optimizerD.pth' % (output_dir))
                   

    return 



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

    output_dir = '../output/%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME)

    log_dir = output_dir + '/log'
    mkdir_p(output_dir)
    mkdir_p(output_dir+'/imgs')
    mkdir_p(output_dir+'/models')
    mkdir_p(log_dir)

    logger = setup_logger(cfg.CONFIG_NAME,log_dir)
    logger.info('Using config:')
    logger.info(str(cfg))

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    logger.info("seed now is : ",str(args.manualSeed))
    
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    #if cfg.CUDA:
    torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        logger.info(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        logger.info(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).cuda()
    netD = NetProjD_V2().cuda()

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch = args.resume_epoch

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    if state_epoch != 0:
        netG.load_state_dict(torch.load('%s/models/netG_%03d.pth' % (output_dir, state_epoch),map_location='cpu'))
        netD.load_state_dict(torch.load('%s/models/netD_%03d.pth' % (output_dir, state_epoch),map_location='cpu'))
        netG = netG.cuda()
        netD = netD.cuda()
        optimizerG.load_state_dict(torch.load('%s/models/optimizerG.pth' %(output_dir)))
        optimizerD.load_state_dict(torch.load('%s/models/optimizerD.pth' % (output_dir)))

    netG.cuda()
    netD.cuda()

    if cfg.B_VALIDATION:
        sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        logger.info('state_epoch:  %d'%(state_epoch))
    else:
        train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device,output_dir,logger)



        