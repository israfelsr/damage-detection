# Common imports
from functools import partial
import random
# Torch
from torch.utils.data import DataLoader

# Local imports
from classifiers.models import FCN
from utils.metrics import IoUMetric

# local imports
from datasets.dataloader import *

if __name__ == '__main__':

    config = {
        "TRAIN_BATCH_SIZE" : 8
    }

    # Set random seed
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    # Create the dataloaders
    train_gen, val_gen, test_gen = load_datasets()
    train_loader = DataLoader(train_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                            shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_gen, batch_size=config["TRAIN_BATCH_SIZE"],
                             shuffle=False, num_workers=1, drop_last=True)
    dataloaders = {
        "train" : train_loader,
        "val" : val_loader,
        "test" : test_loader
    }

    print("len train data:", len(dataloaders["train"]))
    print("len val data:", len(dataloaders["val"]))
    print("len test data:", len(dataloaders["test"]))
    
    # Testing Model
    
    # Testing Metrics
    iou = IoUMetric()
    #sample = dataloaders["train"]
    for t, batch in enumerate(dataloaders["train"]):
        x = batch["X"]
        print("x", x.shape)
        y = batch["y"]
        print(y.shape)
        if t == 0: break

    a = torch.tensor(np.random.rand(3,4))
    b = torch.tensor([[0, 1, 0, 0],[0,1,0,1], [0, 0, 0, 1]], dtype=torch.int8)
    th = np.arange(0.5, 0.95, 0.05)
    partial_iou = torch.zeros((b.shape[0]))
    for thr in th:
        b_pred = (a > thr).type(torch.int8)
        tp = torch.sum(torch.logical_and(b_pred[:,]==1, b[:,]==1), axis=1)
        fp = torch.sum(torch.logical_and(b_pred[:,]==1, b[:,]==0), axis=1)
        fn = torch.sum(torch.logical_and(b_pred[:,]==0, b[:,]==1), axis=1)
        partial_iou += tp/(tp + fp + fn)
        print("partial iou", partial_iou)
    print("-"*5)
    print("total iou", partial_iou/len(th))
    print("avg", torch.mean(partial_iou/len(th)))
    iou.update(a,b)
    print("total iou", iou.get())
    