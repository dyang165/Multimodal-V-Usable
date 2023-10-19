import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import pickle
import json
import os
import tqdm
import numpy as np
import librosa
torch.random.manual_seed(42)

class AVMNIST(Dataset):
    def __init__(self, split, dropout_p=None, audiodir='/home1/dyang165/Datasets/avmnist_noise/audio/wavs_16k', imagedir='/home1/dyang165/Datasets/avmnist_noise/image', labeldir='/home1/dyang165/Datasets/avmnist_noise'):
    # def __init__(self, split, dropout_p=None, audiodir='/scratch2/jihwan/Data/avmnist/audio_temp/wavs_16k', imagedir='/scratch2/jihwan/Data/avmnist/image_nopca', labeldir='/scratch2/jihwan/Data/avmnist'):

        if split == 'dev':
            split = 'new_val'
        elif split == 'test':
            split = 'new_test'

        self.audio_data_dir = audiodir
        self.image_data = np.load(os.path.join(imagedir,f'{split}_data.npy'))
        self.label_data = np.load(os.path.join(labeldir,f'{split}_labels.npy'))
        
        self.num_classes = 10
        self.split = split
        

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, x):
        audio, sr = librosa.load(os.path.join(self.audio_data_dir, f'{self.split}_wav_{x}.wav'), sr=16000) #, self.audio_data[x]
        image = self.image_data[x].reshape((28,28,1)).astype(np.uint8)
        image = np.repeat(image, 3, axis=-1)
        # image = self.image_data[x]
        # print(imag[[e)
        # image = torch.Tensor(image)
        # image = np.expand_dims(image, axis=-1)
        # print(image.shape)
        label = self.label_data[x]
        return {'audio':audio, 'image':image, 'labels':label}

    
if __name__ == '__main__':
    db = AVMNIST("train")
    # print(db.__getitem__(0))
    # print(db.__getitem__(0)['image'])
    temp_img = db.__getitem__(3)['image'][:,:,0]
    # temp_img = db.__getitem__(4)['image']
    print(db.__getitem__(3)['labels'])
    print(temp_img.shape)
    import soundfile as sf
    sf.write('temp.wav', db.__getitem__(3)['audio'], 16000)

    import matplotlib.pyplot as plt
    plt.imshow(temp_img, cmap='gray')
    plt.legend()
    plt.savefig('./temp.jpg')

    temp_arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    temp_arr = np.reshape(temp_arr, 9)
    temp_arr = np.reshape(temp_arr, (3,3))
    print(temp_arr)
