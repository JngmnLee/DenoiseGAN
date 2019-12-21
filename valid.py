import numpy as np
import argparse
import cv2
import torch
import os
from skimage.measure import compare_ssim


def ssim(img1, img2):
    if img1.shape != img2.shape:
        raise Exception

    ss = 0
    # for i in range(img1.shape[0]):
    #     ss += compare_ssim(img1[i, :, :], img2[i, :, :])
    ss = compare_ssim(img1,img2,multichannel=True, win_size=11, gaussian_weights=True,K1=0.01,K2=0.03,sigma=1.5)
    return ss


def val(model, folder):
    test_list = './DB/Denoise_BSD68_test_list.txt'

    sigma = 25
    cnt = 0
    with open(test_list,'r') as f:
        loss_mse = 0
        loss_psnr = 0
        loss_ssim = 0
        for line in f:
            cnt += 1
            clean = cv2.imread(line[:-1])
            test = cv2.imread(line[:-1])
            noisy = np.random.normal(loc=0.0, scale=sigma, size=clean.shape)
            noisy = clean + noisy
            noisy[noisy < 0] = 0
            noisy[noisy > 255] = 255
            H, W, _ = clean.shape
            if H % 2 == 1:
                H -= 1
            if W % 2 == 1:
                W -= 1

            clean = torch.from_numpy(clean.transpose((2, 0, 1)))
            noisy  = torch.from_numpy(noisy.transpose((2, 0, 1)))
            clean = clean.unsqueeze(0).float()
            noisy = noisy.unsqueeze(0).float()/255
            clean = clean[:, :, :H, :W].cuda()
            noisy = noisy[:, :, :H, :W].cuda()
            pred = model(noisy)*255

            pred_numpy = pred.data[0].cpu().numpy()
            clean_numpy = clean.data[0].cpu().numpy()
            pred_numpy = np.clip(pred_numpy, 0, 255)

            pred_numpy = pred_numpy.transpose(1, 2, 0).astype("uint8")
            target_numpy = clean_numpy.transpose(1, 2, 0).astype("uint8")

            loss = np.mean((pred_numpy - target_numpy) ** 2)
            loss_mse += loss
            loss_psnr += 10 * np.log10(255 * 255 / loss)
            ssim_tmp = ssim(pred_numpy, target_numpy)
            loss_ssim += ssim_tmp
            cv2.imwrite(folder+'%d.jpg'%(cnt), pred_numpy)
            #print(cnt, '/68', 'PSNR: ', 10 * np.log10(255 * 255 / loss), 'SSIM: ', ssim_tmp)
    print('MSE: %f, psnr: %f, ssim: %f' %(loss_mse/68, loss_psnr/68, loss_ssim/68))