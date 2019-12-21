import data
import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from valid import val


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="Image Denoise GAN")
parser.add_argument('--dataset', default='bsd_waterloo')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int, help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=0.0002, type=float, help='initial learning rate')
parser.add_argument('--b1', type=float, default=0.0)
parser.add_argument('--b2', type=float, default=0.9)
parser.add_argument('--save_folder', default='weights/GeneratorMk2/', help='Directory for saving checkpoint models')
parser.add_argument('--max_epoch', default=100, type=int, help='Max epoch')
parser.add_argument('--print_freq', default=100, type=int, help='Display frequency')
parser.add_argument('--operation', default='Denoise')
parser.add_argument('--eval_freq', default=1, type=int)
args = parser.parse_args()

from model import Generator, Discriminator


if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING!!!!: \n It is not a CUDA device")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
if not os.path.exists(args.save_folder + 'DEMO/'):
    os.mkdir(args.save_folder + 'DEMO/')

for i in range(args.max_epoch):
    if not os.path.exists(args.save_folder + 'DEMO/' + '%d/'%(i+1)):
        os.mkdir(args.save_folder + 'DEMO/' + '%d/'%(i+1))


class Average(object):
    def __init__(self):
        self.num = 0
        self.count = 0

    def add(self, num):
        self.num += num
        self.count += 1

    def pop(self):
        if self.count != 0:
            return self.num / self.count
        else:
            raise ValueError

    def reset(self):
        self.num = 0
        self.count = 0


def train():

    G = Generator()
    D = Discriminator()

    if args.cuda:
        G.cuda()
        D.cuda()

    criterion = torch.nn.MSELoss()
    criterion = criterion.cuda()
    L1 = torch.nn.L1Loss().cuda()
    G.train()
    D.train()
    print(G,'\n',D)

    train_dataset = data.ArbitraryImageFolder('DB','_train_data_ori.npy', 25)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=data.var_custom_collate, num_workers=args.num_workers,
                                  pin_memory=False, drop_last=False)
    #train_dataloader = data2.train_dataloader('/hdd/dataset/coex/bsd_waterloo/waterloo/', batch_size=args.batch_size, num_workers=args.num_workers)
    optimizer_G = torch.optim.Adam(G.parameters(), 0.0001, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(D.parameters(), 0.0004, betas=(args.b1, args.b2))

    iters = len(train_dataloader)

## Train
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G_EP = Average()
    D_EP = Average()
    for ep in range(args.start_epoch, args.max_epoch):
        G_iter = Average()
        D_iter = Average()
        tm = time.time()
        for curr_iter, batch_data in enumerate(train_dataloader):
            noisy, clean = batch_data
            noisy = noisy.cuda()
            clean = clean.cuda()

            valid = Tensor(noisy.shape[0], 1 ).fill_(1.0).detach()
            fake = Tensor(noisy.shape[0], 1).fill_(0.0).detach()

            optimizer_D.zero_grad()
            gen_img = G(noisy)
            real_loss = criterion(D(clean), valid)
            fake_loss = criterion(D(gen_img.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            g_loss = criterion(D(gen_img), valid)
            L1_loss = L1(gen_img, clean)
            total_loss = L1_loss * 10 + g_loss
            total_loss.backward()
            optimizer_G.step()

            G_EP.add(g_loss)
            D_EP.add(d_loss)
            G_iter.add(g_loss)
            D_iter.add(d_loss)

            if (curr_iter+1) % args.print_freq == 0:
                print('%d/%d Loss G: %8.5f Loss D: %8.5f'%(curr_iter, iters, G_iter.pop(), D_iter.pop()))
                G_iter.reset()
                D_iter.reset()
        print('EPOCH: %d'%(ep + 1))
        print('Loss G: %8.5f Loss D: %8.5f'%(G_EP.pop(), D_EP.pop()))
        print('Time: %f'%(( time.time()-tm)/60))
        torch.save(G.state_dict(), args.save_folder + 'G_%d'%(ep))
        torch.save(D.state_dict(), args.save_folder + 'D_%d' % (ep))

        if ep % args.eval_freq == 0:
            G.eval()
            with torch.no_grad():
                val(G, args.save_folder + 'DEMO/' + '%d/'%(ep+1))
            G.train()


if __name__ == '__main__':
    print(torch.__version__)
    print(args)
    train()