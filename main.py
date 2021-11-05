# torch
from torch.utils.data import DataLoader

# local imports
from datasets.dataloader import *

if __name__ == '__main__':

    config = {
        "TRAIN_BATCH_SIZE" : 8
    }

    # Create the dataloaders
    train_gen, val_gen, test_gen = load_datasets()
    train_loader = DataLoader(train_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                              shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                            shuffle=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                             shuffle=True, num_workers=8, drop_last=True)
    dataloaders = {
        "train" : train_loader,
        "val" : val_loader,
        "test" : test_loader
    }
    print(len(dataloaders["test"]))
    print(len(dataloaders["train"]))
    print(len(dataloaders["val"]))
    


