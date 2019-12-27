from dataset import *
from cnn4conv import *

import sys,os
import time
import argparse
import torch
import torch.backends.cudnn
import torch.optim

from torch.utils.data import DataLoader
from typing import Union, NamedTuple
from multiprocessing import cpu_count


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a Two-Stream CNN on Urbansound8K",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--learning-rate", default=1e-2, type=float, help="Learning rate")
parser.add_argument("--mode", default="LMC", type=str, help="LMC or MC")
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


    model = CNN(1, 10, 0.5)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0004)
    
    val_frequency = 5
    step = 0
    print_frequency = 500

    for epoch in range(0, 50):
        model.train()
   
        for batch, target, filename in train_loader:
            model.train()

            batch = batch.to(DEVICE)
            target = target.to(DEVICE)
            data_load_end_time = time.time()
            optimizer.zero_grad()

            logits = model.forward(batch)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            
            
            with torch.no_grad():
                preds = logits.argmax(-1)
                accuracy = compute_accuracy(target, preds)

            step_time = time.time() - data_load_end_time

            if ((step + 1) % print_frequency) == 0:
                print_metrics(step, epoch, accuracy, loss, step_time, train_loader)
            step += 1
            
        
        if ((epoch + 1) % val_frequency) == 0:
            all_logits = torch.zeros((0,10), dtype=torch.float32).cuda()
            all_targets = []
            all_filenames = []
            list_filenames = []
            # results = {"preds": [], "labels": []}
            average_logits = torch.zeros((0, 10), dtype=torch.float32).cuda()           
            new_targets = []
            total_loss = 0
            model.eval()

            with torch.no_grad():
                for batch, target, filename in val_loader:
                    batch = batch.to(DEVICE)
                    target = target.to(DEVICE)
                    logits = model(batch)
                    loss = criterion(logits, target)
                    total_loss += loss.item()

                    all_logits = torch.cat((all_logits, logits))
                    all_targets.extend(list(target.cpu().numpy()))
                    all_filenames.extend(filename)

                    for unique_file_name in filename:
                        if unique_file_name not in list_filenames:
                            list_filenames.append(unique_file_name)

                    # preds = logits.argmax(dim=-1).cpu().numpy()
                    # results["preds"].extend(list(preds))
                    # results["labels"].extend(list(target.cpu().numpy()))


            for name in list_filenames:
                temp_list = [name] * len(all_filenames)
                bool_list = [False] * len(all_filenames)
                temp_index = 0
                for index, item in enumerate(all_filenames):
                    if temp_list[index] == item:
                        bool_list[index] = True
                        temp_index = index
                average_logits = torch.cat((average_logits, all_logits[bool_list].mean(dim=0, keepdim=True)))
                new_targets.append(all_targets[temp_index])
              
            predicts = average_logits.argmax(dim=-1).cpu().numpy()

         
            # results["preds"].extend(list(preds))

                
            # accuracy = compute_accuracy(
            #     np.array(results["labels"]), np.array(results["preds"])
            # )

            accuracy = compute_accuracy(np.array(new_targets), np.array(predicts))
            average_loss = total_loss / len(val_loader)

            
            # class_accuracy = compute_class_accuracy(
            #     np.array(results["labels"]), np.array(results["preds"])
            # )
            class_accuracy = compute_class_accuracy(np.array(new_targets), np.array(predicts))

            print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")
            print("per-class accuracy: {}".format(class_accuracy))

            model.train()

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

  
    for index, class_number in enumerate(labels):     
        classes[class_number] = classes[class_number] + 1
        if class_number == preds[index]:
            correct_classes[class_number] = correct_classes[class_number] + 1
       

    result = correct_classes / classes * 100
    return result


def print_metrics(step, epoch, accuracy, loss, step_time, train_loader):
        epoch_step = step % len(train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                # f"data load time: "
                # f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
    

if __name__ == "__main__":
    main(parser.parse_args())