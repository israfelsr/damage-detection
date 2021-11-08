import numpy as np
from sklearn.model_selection import train_test_split

# torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# local imports
from utils.data_utils import load_data

class MaterialData(Dataset):
    def __init__(self, X_data, y_data):
        super(MaterialData, self).__init__()
        self.imgs = X_data
        self.masks = y_data
        self.tfrm = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index,:,:]
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
        mask = self.masks[index,:,:]
        img = self.tfrm(img)
        mask = self.tfrm(mask)
        return {"X" : img, "y" : mask}

def load_datasets():
    filename = 'datasets/Deltas3.mat'
    X_train, X_test, y_train, y_test = load_data(filename)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.20, 
                                                      random_state=42)
    train_gen = MaterialData(X_train, y_train)
    val_gen = MaterialData(X_val, y_val)
    test_gen = MaterialData(X_test, y_test)
    return train_gen, val_gen, test_gen
    

    