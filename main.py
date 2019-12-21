from dataset import *
from cnn4conv import *

import time
import argparse
import torch
import torch.backends.cudnn
from torch.utils.data import DataLoader
from typing import Union, NamedTuple


torch.backends.cudnn.benchmark = True

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
        UrbanSound8KDataset('UrbanSound8K_test.pkl', 'LMC'), 
        batch_size=32, shuffle=False, 
        num_workers=8, pin_memory=True) 

    model = CNN(1, 10, 0.5)
    criterion = nn.Softmax()
    model.to(DEVICE)
    val_frequency = 2
    step = 0
    print_frequency = 20

    for epoch in range(0, 50):
        model.train()

        data_load_start_time = time.time()
        for i, (batch, target, filename) in enumerate(train_loader): 
            batch = batch.to(DEVICE)
            target = target.to(DEVICE)
            data_load_end_time = time.time()

            logits = model.forward(batch)
            loss = criterion(logits, target)
            loss.backward()

            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = compute_accuracy(target, preds)

            data_load_time = data_load_end_time - data_load_start_time
            step_time = time.time() - data_load_end_time

            if ((step + 1) % print_frequency) == 0:
                print_metrics(step, epoch, accuracy, loss, data_load_time, step_time, train_loader)
            step += 1
            data_load_start_time = time.time()

        if ((epoch + 1) % val_frequency) == 0:
            results = {"preds": [], "labels": []}
            total_loss = 0
            model.eval()

            with torch.no_grad():
                for batch, target in val_loader:
                    batch = batch.to(DEVICE)
                    target = target.to(DEVICE)
                    logits = model(batch)
                    loss = criterion(logits, target)
                    total_loss += loss.item()
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    results["preds"].extend(list(preds))
                    results["labels"].extend(list(target.cpu().numpy()))

            accuracy = compute_accuracy(
                np.array(results["labels"]), np.array(results["preds"])
            )
            average_loss = total_loss / len(val_loader)

            class_accuracy = compute_class_accuracy(
                np.array(results["labels"]), np.array(results["preds"])
            )
            
            print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
            print("per-class accuracy: {}".format(class_accuracy))


def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_class_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> np.array:

    assert len(labels) == len(preds)
    classes = np.zeros(10)
    correct_classes = np.zeros(10)

    a = 0
    for class_number in labels:     
        classes[class_number] = classes[class_number] + 1
        if class_number == preds[a]:
            correct_classes[class_number] = correct_classes[class_number] + 1
        a = a + 1

    result = correct_classes / classes * 100
    return result


def print_metrics(step, epoch, accuracy, loss, data_load_time, step_time, train_loader):
        epoch_step = step % len(train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )


if __name__ == "__main__":
    main(parser.parse_args())