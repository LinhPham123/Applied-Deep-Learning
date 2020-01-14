from dataset import *
from utils import *
from train import *
from cnnNoStride import *
from doubleTrain import *

import sys,os
from pathlib import Path
import argparse
from multiprocessing import cpu_count

import torch
import torch.backends.cudnn
import torch.optim
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a Two-Stream CNN on Urbansound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--mode", default="LMC", type=str, help="LMC, MC, MLMC or TSCNN")
parser.add_argument("--L2", default=1e-3, type=float)
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--epochs", default=100, type=int)


parser.add_argument("--checkpoint-path", default=Path("saved_model.pt"), type=Path)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

dirname = os.path.dirname(__file__)
my_train = os.path.join(dirname, 'UrbanSound8K_train.pkl')
my_test = os.path.join(dirname, 'UrbanSound8K_test.pkl')


def main(args):
    height = 85
    width = 41
    channels = 1
    if args.mode == "MLMC":
        height = 145

    criterion = nn.CrossEntropyLoss()

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    if args.mode in ['LMC', 'MC', 'MLMC']:
        train_dataset = UrbanSound8KDataset(my_train, args.mode)
        test_dataset = UrbanSound8KDataset(my_test, args.mode)
    
        train_loader = torch.utils.data.DataLoader( 
            train_dataset, 
            batch_size=32, shuffle=True, 
            num_workers=cpu_count(), pin_memory=True)
 
        val_loader = torch.utils.data.DataLoader( 
            test_dataset, 
            batch_size=32, shuffle=False,
            num_workers=cpu_count(), pin_memory=True) 


        model = CNNNoStride(input_height=height,input_width=width,input_channels=channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.L2)
        trainer = Trainer(model=model, device=DEVICE, 
                    train_loader=train_loader, val_loader=val_loader, 
                    criterion=criterion, optimizer=optimizer, 
                    summary_writer=summary_writer, epochs=args.epochs)
        trainer.train()


    else:
        train_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 UrbanSound8KDataset(my_train, "LMC"),
                 UrbanSound8KDataset(my_train, "MC")
             ),
             batch_size=32, shuffle=True,
             num_workers=cpu_count(), pin_memory=True) 

        val_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 UrbanSound8KDataset(my_test, "LMC"),
                 UrbanSound8KDataset(my_test, "MC")
             ),
             batch_size=32, shuffle=False,
             num_workers=cpu_count(), pin_memory=True)

 
        LMCNet = CNNNoStride(input_height=height,input_width=width,input_channels=channels)
        MCNet = CNNNoStride(input_height=height,input_width=width,input_channels=channels)

        LMC_optimizer = torch.optim.Adam(LMCNet.parameters(), lr=args.learning_rate, weight_decay=args.L2)
        MC_optimizer = torch.optim.Adam(MCNet.parameters(), lr=args.learning_rate, weight_decay=args.L2)

        double_trainer = DoubleTrainer(LMCNet, MCNet, DEVICE, train_loader, val_loader, criterion, LMC_optimizer, MC_optimizer, summary_writer, args.epochs)
        double_trainer.train()
   
    summary_writer.close()
 

if __name__ == "__main__":
    main(parser.parse_args())