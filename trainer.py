import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time

class Trainer:
    def __init__(self, model: nn.Module, device: torch.device,
                train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: Optimizer,
                summary_writer: SummaryWriter, epochs: int = 100):
                
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.epochs = epochs
        self.step = 0

    def train(self, val_frequency: int = 2, print_frequency: int = 1000, log_frequency: int = 500):
            self.model.train()
            for epoch in range(self.epochs):
                self.model.train()       
                for batch, labels, _ in self.train_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    
                    logits = self.model.forward(batch)
                    loss = self.criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    with torch.no_grad():
                        preds = logits.argmax(-1)
                        batch_class_accuracies = compute_class_accuracy(labels, preds)
                        average_class_accuracy = batch_class_accuracies.sum() / 10

                    if ((self.step + 1) % log_frequency) == 0:
                        self.log_trainning_metrics(average_class_accuracy, loss)
                    if ((self.step + 1) % print_frequency) == 0:
                        self.print_training_metrics(average_class_accuracy, loss)

                    self.step += 1

                if ((epoch + 1) % val_frequency) == 0:
                    self.validate()
                    self.model.train()
                     
    def validate(self):
        total_loss = 0
        self.model.eval()
        results = {"preds": [], "labels": []}

        file_logits = torch.Tensor([]).to(self.device) #Each row is the logits of 1 unique file
        unique_filenames = {} #Each key: a unique filename. Each key's value: the index of the row corresponding to that file in file_logits
        unique_filenames_count = 0

        with torch.no_grad():
            for batch, labels, batch_filenames in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()
               
                for i, filename in enumerate(batch_filenames):
                    if filename not in unique_filenames.keys():
                        unique_filenames[filename] = unique_filenames_count
                        file_logits = torch.cat((file_logits, logits[i:i+1]), 0) #Logits[i] return 1D, logits[i:i+1] return 2D
                        results["labels"].append(labels[i].cpu().numpy()) #All segements of a file belong to a same class                       
                        unique_filenames_count += 1
                    else:
                        file_logits[unique_filenames[filename]] += logits[i]

            preds = file_logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))

       
        average_loss = total_loss / len(self.val_loader)

        class_accuracies = compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_class_accuracy = class_accuracies.sum() / 10

        print(f"validation loss: {average_loss:.5f}, average class accuracy: {average_class_accuracy}")
        print(f"per-class accuracy: {class_accuracies}")

        self.summary_writer.add_scalars(
                "average accuracy",
                {"test": average_class_accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )

    def print_training_metrics(self, accuracy, loss):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{self.epochs}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy}, "
        )

    def log_trainning_metrics(self, accuracy, loss):
        self.summary_writer.add_scalars(
                "average accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
    
