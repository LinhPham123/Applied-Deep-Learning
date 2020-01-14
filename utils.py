from typing import Union, NamedTuple
import torch
import numpy as np
import argparse

def calculate_padding(input: int, output: int, filter: int, stride: int):
    return int(np.ceil(((output - 1) * stride - input + filter) / 2))

def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> np.array:
    assert len(labels) == len(preds)
    classes = np.zeros(10)
    correct_classes = np.zeros(10)

    for index, class_number in enumerate(labels):     
        classes[class_number] = classes[class_number] + 1
        if class_number == preds[index]:
            correct_classes[class_number] = correct_classes[class_number] + 1
    
    return np.divide(correct_classes, classes, out=np.full(correct_classes.shape, fill_value=1, dtype=float), where=classes!=0) * 100
    

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
        f"CNN_bn_"
        f"mode={args.mode}_"
        # f"bs={32}_"
        f"L2={args.L2}_"
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)    


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
