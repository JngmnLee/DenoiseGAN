import numpy as np
import torch
import torch.utils.data as data


def var_custom_collate(batch):

    batch_input_images = torch.Tensor(len(batch), 3, 64, 64)
    batch_target_images = torch.Tensor(len(batch), 3, 64, 64)

    for idx, item in enumerate(batch):
        batch_input_images[idx, 0:3] = item['noisy'][:,:,:64,:64]/255.0
        batch_target_images[idx] = item['input'][:,:,:64,:64]/255.0
    return batch_input_images, batch_target_images


class ArbitraryImageFolder(data.Dataset):

    def __init__(self, path, input_name, sigma):
        self.path = path
        self.input_name = input_name

        self.sigma = sigma

        self.inputs = []
        for i in range(14):
            self.inputs.append(np.load(self.path + '/' + '{0:02d}'.format(i) + self.input_name))

    def __getitem__(self, index):

        # input_name = self.path + '/' + '{0:02d}'.format(index//6000) + self.input_name
        # self.input_paths = np.load(input_name)
        #input = self.input_paths[index % 6000].transpose((2, 0, 1))

        input = self.inputs[index // 6000][index % 6000].transpose((2, 0, 1))

        noisy = np.random.normal(loc=0.0, scale=self.sigma, size=input.shape)
        noisy = input + noisy
        noisy[noisy<0] = 0
        noisy[noisy>255] =255
        input = torch.from_numpy(input).unsqueeze(0).float()
        noisy = torch.from_numpy(noisy).unsqueeze(0).float()

        return {'input': input, 'noisy': noisy}

    def __len__(self):
        return 6000 * 14

    def getimg(self, idex):
        pass
