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
                summary_writer: SummaryWriter):
                
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0


    def train(self, epochs: int = 50, val_frequency: int = 5, print_frequency: int = 1000, log_frequency: int = 5):
            self.model.train()
            for epoch in range(epochs):
                self.model.train()
                data_load_start_time = time.time()

                for batch, labels, _ in self.train_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    data_load_end_time = time.time()

                    logits = self.model.forward(batch)
                    loss = self.criterion(logits, labels)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    with torch.no_grad():
                        preds = logits.argmax(-1)
                        batch_accuracy = compute_accuracy(labels, preds)

                    data_load_time = data_load_end_time - data_load_start_time
                    step_time = time.time() - data_load_end_time

                    if ((self.step + 1) % log_frequency) == 0:
                        self.log_trainning_metrics(epoch, batch_accuracy, loss, data_load_time, step_time)
                    if ((self.step + 1) % print_frequency) == 0:
                        self.print_training_metrics(epoch, batch_accuracy, loss, data_load_time, step_time)

                    self.step += 1
                    data_load_start_time = time.time()

                # self.summary_writer.add_scalar("epoch", epoch, self.step)
                
                if ((epoch + 1) % val_frequency) == 0:
                    self.validate()
                    self.model.train()


    def validate(self):
        total_loss = 0
        self.model.eval()

        all_logits = torch.zeros((0,10), dtype=torch.float32).cuda()
        all_labels = []
        all_filenames = []
        list_unique_filenames = []
    
        average_logits = torch.zeros((0, 10), dtype=torch.float32).cuda()           
        new_targets = []          

        with torch.no_grad():
            for batch, labels, batch_filenames in self.val_loader:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(batch)
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                    all_logits = torch.cat((all_logits, logits))
                    all_labels.extend(list(labels.cpu().numpy()))
                    all_filenames.extend(batch_filenames)

                    for unique_file_name in batch_filenames:
                        if unique_file_name not in list_unique_filenames:
                            list_unique_filenames.append(unique_file_name)

                 
            for name in list_unique_filenames:
                temp_list = [name] * len(all_filenames)
                bool_list = [False] * len(all_filenames)
                temp_index = 0
                for index, item in enumerate(all_filenames):
                    if temp_list[index] == item:
                        bool_list[index] = True
                        temp_index = index
                average_logits = torch.cat((average_logits, all_logits[bool_list].mean(dim=0, keepdim=True)))
                new_targets.append(all_labels[temp_index])
              
            predicts = average_logits.argmax(dim=-1).cpu().numpy()
    

        accuracy = compute_accuracy(np.array(new_targets), np.array(predicts))
        average_loss = total_loss / len(self.val_loader)
        class_accuracy = compute_class_accuracy(np.array(new_targets), np.array(predicts))

        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
        print("per-class accuracy: {}".format(class_accuracy))

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )


    def print_training_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
            f"epoch: [{epoch}], "
            f"step: [{epoch_step}/{len(self.train_loader)}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
            f"data load time: "
            f"{data_load_time:.5f}, "
            f"step time: {step_time:.5f}"
        )


    def log_trainning_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        # self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        # self.summary_writer.add_scalar(
        #         "time/data", data_load_time, self.step
        # )
        # self.summary_writer.add_scalar(
        #         "time/data", step_time, self.step
        # )        