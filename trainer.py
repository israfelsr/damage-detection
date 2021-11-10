import wandb
import time
import numpy as np

import torch

class Trainer:
    def __init__(self, model, device, loss_metric, score_metric, criterion, optimizer):
        self.model = model
        self.device = device
        self.loss_metric = loss_metric
        self.score_metric = score_metric
        self.criterion = criterion
        self.optimizer = optimizer
        self.messages = {
            "epoch" : "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint" : "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
        }
        self.total_samples_trained_on = 0
        self.best_valid_score = -np.inf
        self.dummy_input = None
    
    def fit(self, epochs, dataloaders, save_path):
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        for epoch in range(epochs):
            train_loss, train_score, train_time = self.run_epoch(epoch, dataloaders, mode="train")
            val_loss, val_score, val_time = self.run_epoch(epoch, dataloaders, mode="val")

            if val_score > self.best_valid_score:
                self.save_model(epoch, save_path)
        
    def run_epoch(self, epoch, dataloaders, mode):
        is_train = mode == "train"
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        
        t = time.time()
        running_loss = self.loss_meter()
        running_score = self.score_meter()
        
        for _, batch in enumerate(dataloaders["mode"]):
            with torch.set_grad_enabled(is_train):
                X = batch["X"].to(self.device)
                y_true = batch["y"].to(self.device)
                if is_train:
                    self.optimizer.zero_grad()
                if self.dummy_input == None:
                    self.dummy_input = X
                
                outputs = self.model(X)
                loss = self.criterion(outputs["out"], y_true)
                
                if is_train:
                    loss.backwards()
                running_loss.update(loss.detach().item())
                running_score.update(outputs["out"].detach(), y_true)
                #if self.device=='cuda':
                #    current_score.update(outputs.detach(), y_true)
                #else:
                #    current_score.update(outputs.detach(), y_true)
                _loss, _score = running_loss.get(), running_score.get()
                if is_train:
                    self.total_samples_trained_on += X.shape[0]
                    wandb.log({
                        "epoch" : epoch, "loss" : _loss, "train_score": _score},
                        step=self.total_samples_trained_on)
        if not is_train:
            wandb.log({
                "epoch": epoch, "validation_score": running_score.get()},
                 step=self.total_examples_trained_on)
        return running_loss.get(), running_score.get(), int(time.time() - t)

    def save_model(self, n_epoch, save_path):
        torch.save(
            {"model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "best_valid_score": self.best_valid_score,
            "n_epoch" : n_epoch},
            save_path
        )
        torch.onnx.export(self.model, self.dummy_input, "model.onnx")
        wandb.save("model.onnx")