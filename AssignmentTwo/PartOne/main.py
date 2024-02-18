import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch

from sklearn.metrics import accuracy_score
from utils.general_utils import (
    create_helper_directories,
    evaluate_model,
    load_dataset,
    prepare_model_for_evaluation,
    prepare_model_for_training,
)
from utils.model import VGG11
from torch.nn.functional import log_softmax

torch.set_num_threads(4)
torch.manual_seed(42)

def train_model(
    model,
    device,
    criterion,
    optimizer,
    start_epoch,
    end_epoch,
    train_loader,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )
    model.train()

    epoch_duration_list = []

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss_list = []
        train_accuracy_list = []
        train_duration_list = []

        train_loss = 0.0
        train_accuracy = 0.0
        avg_batch_duration = 0.0

        start_time = time.time()
        end_time = 0.0
        epoch_duration = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            y_hat = model(data)
            loss = criterion(y_hat, target)
            batch_loss = loss.item()
            loss.backward()
            optimizer.step()

            y_pred = log_softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().tolist()
            target = target.detach().cpu().tolist()

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            batch_accuracy = accuracy_score(target, y_pred)

            train_loss_list.append(batch_loss)
            train_accuracy_list.append(batch_accuracy)
            train_duration_list.append(batch_duration)

            if batch_idx % 20 == 0:
                print(
                    "Batch Idx: {}, Batch Loss: {:.3f}, Batch Accuracy: {:.3f}, Batch Duration: {:.3f} seconds".format(
                        batch_idx, batch_loss, batch_accuracy, batch_duration
                    )
                )
                print("--------------------")

            torch.cuda.empty_cache()

            del data, target
            del y_hat, loss, y_pred
            del batch_loss, batch_accuracy
            del batch_start_time, batch_end_time

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_duration_list.append(epoch_duration)

        train_loss = sum(train_loss_list) / len(train_loss_list)
        train_accuracy = sum(train_accuracy_list) / len(train_accuracy_list)
        avg_batch_duration = sum(train_duration_list) / len(train_duration_list)

        with open(logs_path, "at") as logs_file:
            logs_file.write(
                "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Epoch Duration: {:.3f} seconds\n".format(
                    epoch,
                    train_loss,
                    train_accuracy,
                    avg_batch_duration,
                    epoch_duration,
                )
            )
        print(
            "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Epoch Duration: {:.3f} seconds".format(
                epoch, train_loss, train_accuracy, avg_batch_duration, epoch_duration
            )
        )
        print("--------------------")

        ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, str(epoch))
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            ckpt_path,
        )
        del train_loss_list, train_accuracy_list, train_duration_list
        del train_loss, train_accuracy, avg_batch_duration
        del start_time, end_time, epoch_duration

    avg_epoch_duration = sum(epoch_duration_list) / len(epoch_duration_list)
    print("Avg Epoch Duration: {:.3f} seconds".format(avg_epoch_duration))
    print("--------------------")
    del avg_epoch_duration, epoch_duration_list


def main():
    print(
        "\n--------------------\nStarting model training and evaluation!\n--------------------\n"
    )
    batch_size = 256
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    start_epoch = 1
    end_epoch = 1
    continue_flag = False
    checkpoint_dir = "Checkpoints"
    logs_dir = "Logs"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available!")
    else:
        device = torch.device("cpu")
        print("GPU not available!")

    model = VGG11()
    model.to(device)

    train_loader, test_loader = load_dataset(batch_size=batch_size)

    checkpoint_dir, logs_path, report_path = create_helper_directories(
        checkpoint_dir=checkpoint_dir, logs_dir=logs_dir, task_name="", flag=False
    )

    model, criterion, optimizer = prepare_model_for_training(
        model=model,
        device=device,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        continue_flag=continue_flag,
    )

    train_model(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        train_loader=train_loader,
        logs_path=logs_path,
        checkpoint_dir=checkpoint_dir,
    )

    checkpoint_path = os.path.join(checkpoint_dir, "Epoch_1.pt")

    model = prepare_model_for_evaluation(
        model=model, device=device, checkpoint_path=checkpoint_path
    )

    evaluate_model(
        model=model,
        device=device,
        criterion=criterion,
        test_loader=test_loader,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )

    print(
        "\n--------------------\nModel training and evaluation complete!\n--------------------\n"
    )


if __name__ == "__main__":
    main()
