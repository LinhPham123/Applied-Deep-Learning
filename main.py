from dataset import UrbanSound8KDataset
import cnn4conv

import argparse
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(
    description="Train a Two-Stream CNN on Urbansound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    train_loader = torch.utils.data.DataLoader( 
        UrbanSound8KDataset('UrbanSound8K_train.pkl', 'LMC'), 
        batch_size=32, shuffle=True, 
        num_workers=8, pin_memory=True) 
 
    val_loader = torch.utils.data.DataLoader( 
        UrbanSound8KDataset('UrbanSound8K_test.pkl', 'MC'), 
        batch_size=32, shuffle=False, 
        num_workers=8, pin_memory=True) 

 

# for i, (input, target, filename) in enumerate(train_loader): 
# #           training code
#     print("hello")

# for i, (input, target, filename) in enumerate(val_loader): 
#     print("hi")
# #           validation code 

 



if __name__ == "__main__":
    main(parser.parse_args())