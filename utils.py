from typing import Union, NamedTuple
import torch
import numpy as np
import argparse


def compute_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> np.array:
    assert len(labels) == len(preds)
    classes = np.zeros(10)
    correct_classes = np.zeros(10)

    for index, class_number in enumerate(labels):     
        classes[class_number] = classes[class_number] + 1
        if class_number == preds[index]:
            correct_classes[class_number] = correct_classes[class_number] + 1
  
    result = correct_classes / classes * 100
    return result


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
    #tb_log_dir_prefix = f'CNN_bn_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.sgd_momentum}_run_'
    tb_log_dir_prefix = (
        f"CNN_bn_"
        # f"dropout={args.dropout}_"
        f"bs={32}_"
        f"lr={1e-3}_"
        f"momentum=0.9_"
        f"L2={args.L2}_"
        # f"rotation={args.data_aug_rotate}_" +
        # ("hflip_" if args.data_aug_hflip else "") +
        f"run_"
    )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)    