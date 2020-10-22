import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from utils import *
import time

class DoubleTrainer:
    def __init__(self, model1: nn.Module, model2:nn.Module, device: torch.device,
                train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer1: Optimizer, optimizer2: Optimizer,
                summary_writer: SummaryWriter, epochs: int = 100):
                
        self.LMC_model = model1.to(device)
        self.MC_model = model2.to(device)
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2

        self.summary_writer = summary_writer
        self.epochs = epochs
        self.step = 0
        self.softmax = nn.Softmax(dim=1)

    def train(self, val_frequency: int = 2, print_frequency: int = 1000, log_frequency: int = 500):
            self.LMC_model.train()
            self.MC_model.train()
            for epoch in range(self.epochs):
                self.LMC_model.train()
                self.MC_model.train()

                for (LMC_data, MC_data) in self.train_loader:
                    LMC_batch, LMC_labels, _ = LMC_data
                    MC_batch, MC_labels , _ = MC_data

                    LMC_batch = LMC_batch.to(self.device)
                    LMC_labels = LMC_labels.to(self.device)
                    MC_batch = MC_batch.to(self.device)
                    MC_labels = MC_labels.to(self.device)
                    
                    LMC_logits = self.LMC_model.forward(LMC_batch)
                    MC_logits = self.MC_model.forward(MC_batch)
                    logits = (self.softmax(LMC_logits) + self.softmax(MC_logits)) / 2

                    LMC_loss = self.criterion(LMC_logits, LMC_labels)
                    LMC_loss.backward()
                    self.optimizer1.step()
                    self.optimizer1.zero_grad()

                    MC_loss = self.criterion(MC_logits, MC_labels)
                    MC_loss.backward()
                    self.optimizer2.step()
                    self.optimizer2.zero_grad()

                    labels = LMC_labels

                    with torch.no_grad():
                        preds = logits.argmax(-1)
                        batch_class_accuracies = compute_class_accuracy(labels, preds)
                        average_class_accuracy = batch_class_accuracies.sum() / 10

                    if ((self.step + 1) % log_frequency) == 0:
                        self.log_trainning_metrics(average_class_accuracy)
                    if ((self.step + 1) % print_frequency) == 0:
                        self.print_training_metrics(average_class_accuracy, epoch)

                    self.step += 1

                if ((epoch + 1) % val_frequency) == 0:
                    self.validate()
                    self.LMC_model.train()
                    self.MC_model.train()
                     
    def validate(self):
        total_loss = 0
        self.LMC_model.eval()
        self.MC_model.eval()
        results = {"preds": [], "labels": []}

        file_logits = torch.Tensor([]).to(self.device) #Each row is the logits of 1 unique file
        unique_filenames = {} #Each key: a unique filename. Each key's value: the index of the row corresponding to that file in file_logits
        unique_filenames_count = 0

        with torch.no_grad():
            for (LMC_data, MC_data) in self.val_loader:
                LMC_batch, LMC_labels, LMC_batch_filenames = LMC_data
                MC_batch, MC_labels , MC_batch_filenames = MC_data

                print(MC_labels == LMC_labels)
                print(LMC_batch_filenames == MC_batch_filenames)


                LMC_batch = LMC_batch.to(self.device)
                MC_batch = MC_batch.to(self.device)

                labels = LMC_labels.to(self.device)
                LMC_logits = self.LMC_model(LMC_batch)
                MC_logits = self.MC_model(MC_batch)
                logits = (self.softmax(LMC_logits) + self.softmax(MC_logits)) / 2

                for i, filename in enumerate(LMC_batch_filenames):
                    if filename not in unique_filenames.keys():
                        unique_filenames[filename] = unique_filenames_count
                        file_logits = torch.cat((file_logits, logits[i:i+1]), 0) #Logits[i] return 1D, logits[i:i+1] return 2D
                        results["labels"].append(labels[i].cpu().numpy()) #All segements of a file belong to a same class                       
                        unique_filenames_count += 1
                    else:
                        file_logits[unique_filenames[filename]] += logits[i]

            preds = file_logits.argmax(dim=-1).cpu().numpy()
            results["preds"].extend(list(preds))

        class_accuracies = compute_class_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_class_accuracy = class_accuracies.sum() / 10

        # print(f"validation loss: {average_loss:.5f}, average class accuracy: {average_class_accuracy}")
        print(f"average class accuracy: {average_class_accuracy}")
        print(f"per-class accuracy: {class_accuracies}")

        self.summary_writer.add_scalars(
                "average accuracy",
                {"test": average_class_accuracy},
                self.step
        )

    def print_training_metrics(self, accuracy, epoch):
        epoch_step = self.step % (len(self.train_loader))
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch accuracy: {accuracy}, "
        )

    def log_trainning_metrics(self, accuracy):
        self.summary_writer.add_scalars(
                "average accuracy",
                {"train": accuracy},
                self.step
        )
