from dataset import *
from cnn4conv import *
from utils import *
from train import *

import sys,os
import argparse
import torch
import torch.backends.cudnn
import torch.optim

from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a Two-Stream CNN on Urbansound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--mode", default="LMC", type=str, help="LMC or MC")
parser.add_argument("--L2", default=0.001, type=float)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

dirname = os.path.dirname(__file__)
my_train = os.path.join(dirname, 'UrbanSound8K_train.pkl')
my_test = os.path.join(dirname, 'UrbanSound8K_test.pkl')


def main(args):
    train_dataset = UrbanSound8KDataset(my_train, args.mode)
    test_dataset = UrbanSound8KDataset(my_test, args.mode)

    train_loader = torch.utils.data.DataLoader( 
        train_dataset, 
        batch_size=32, shuffle=True, 
        num_workers=0, pin_memory=True) # 0 on local, cpu_count() on bc4
 
    val_loader = torch.utils.data.DataLoader( 
        test_dataset, 
        batch_size=32, shuffle=False,
        num_workers=cpu_count(), pin_memory=True) 


    model = CNN(1, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.L2)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    trainer = Trainer(model, DEVICE, train_loader, val_loader, criterion, optimizer, summary_writer)

    trainer.train()

    summary_writer.close()
 

if __name__ == "__main__":
    main(parser.parse_args())