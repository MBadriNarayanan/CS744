import os
import time
import torch
import torch.cuda

import numpy as np
import torch.distributed as dist

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

np.random.seed(42)


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def create_helper_directories(
    checkpoint_dir, logs_dir, task_name, sub_task_name, rank, flag=False
):
    create_directory(directory=checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, task_name)
    create_directory(directory=checkpoint_dir)

    create_directory(directory=logs_dir)
    logs_dir = os.path.join(logs_dir, task_name)
    create_directory(directory=logs_dir)

    if flag:
        checkpoint_dir = os.path.join(checkpoint_dir, sub_task_name)
        logs_dir = os.path.join(logs_dir, sub_task_name)
        create_directory(directory=checkpoint_dir)
        create_directory(directory=logs_dir)

    if rank == "":
        logs_path = os.path.join(logs_dir, "logs.txt".format(rank))
        report_path = os.path.join(logs_dir, "report.txt".format(rank))
    else:
        logs_path = os.path.join(logs_dir, "logs_rank{}.txt".format(rank))
        report_path = os.path.join(logs_dir, "report_rank{}.txt".format(rank))

    print("Checkpoints will be stored at: {}!".format(checkpoint_dir))
    print("Training logs will be stored at: {}!".format(logs_path))
    print("Evaluation report will be stored at: {}!".format(report_path))

    return checkpoint_dir, logs_path, report_path


def distributed_setup(master_ip, rank, world_size):
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = "1024"
    dist.init_process_group(
        "gloo",
        init_method="tcp://{}:{}".format(
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]
        ),
        rank=rank,
        world_size=world_size,
    )
    print("Running Rank: {}!".format(rank))


def generate_eval_report(
    ground_truth, prediction, test_loss, report_path, checkpoint_path
):
    test_accuracy = accuracy_score(ground_truth, prediction)
    report = classification_report(
        ground_truth,
        prediction,
        digits=3,
        zero_division=0,
    )

    matrix = confusion_matrix(ground_truth, prediction)

    with open(report_path, "w") as report_file:
        report_file.write("Metrics for the checkpoint: {}\n".format(checkpoint_path))
        report_file.write(
            "Test Loss: {:.3f}, Test Accuracy: {:.3f}\n".format(
                test_loss, test_accuracy
            )
        )
        report_file.write("Classification Report\n")
        report_file.write("{}\n".format(report))
        report_file.write("Confusion Matrix\n")
        report_file.write("{}\n".format(matrix))
        report_file.write("--------------------\n")

    print("Metrics for the checkpoint: {}".format(checkpoint_path))
    print("Test Loss: {:.3f}, Test Accuracy: {:.3f}".format(test_loss, test_accuracy))
    print("Classification Report")
    print("{}\n".format(report))
    print("Confusion Matrix\n")
    print("{}\n".format(matrix))
    print("--------------------")

    del ground_truth, prediction
    del test_accuracy, report


def evaluate_model(
    model,
    device,
    criterion,
    test_loader,
    report_path,
    checkpoint_path,
):
    ground_truth = []
    prediction = []

    with torch.no_grad():
        model.eval()
        test_loss = 0.0

        for _, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            y_hat = model(data)
            loss = criterion(y_hat, target)
            item_loss = loss.item()

            y_pred = log_softmax(y_hat, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.detach().cpu().tolist()[0]
            target = target.detach().cpu().tolist()[0]

            test_loss += item_loss
            ground_truth.append(target)
            prediction.append(y_pred)

            del data, target
            del y_hat, y_pred
            del item_loss

    generate_eval_report(
        ground_truth=ground_truth,
        prediction=prediction,
        test_loss=test_loss,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: {}".format(params))
    print("--------------------")


def load_dataset(batch_size):
    normalize = Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    transform_train = Compose(
        [RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize]
    )
    transform_test = Compose([ToTensor(), normalize])

    training_set = CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(
        training_set,
        num_workers=2,
        batch_size=batch_size,
        sampler=None,
        shuffle=True,
        pin_memory=True,
    )

    test_set = CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_set, num_workers=2, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    return train_loader, test_loader


def prepare_model_for_training(
    model,
    device,
    learning_rate,
    momentum,
    weight_decay,
    continue_flag,
    continue_checkpoint_path="",
    distribute_flag=False,
):
    criterion = CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    model = model.to(device)
    criterion = criterion.to(device)
    if distribute_flag:
        model = DDP(model)

    if continue_flag:
        print("Model loaded for further training!")
        checkpoint = torch.load(continue_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Prepared model for training!")
    get_model_parameters(model=model)
    return model, criterion, optimizer


def prepare_model_for_evaluation(model, device, checkpoint_path):
    model = model.to(device)
    print("Loaded checkpoint:", checkpoint_path, "for evaluation!")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("--------------------")
    return model


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
    distribute_flag,
    rank="",
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
                if distribute_flag:
                    print("Rank: {}".format(rank), end=" ")
                print(
                    "Batch Idx: {}, Batch Loss: {:.3f}, Batch Accuracy: {:.3f}, Batch Duration: {:.3f} seconds".format(
                        batch_idx, batch_loss, batch_accuracy, batch_duration
                    )
                )
                print("--------------------")

            if batch_idx == 40:
                if distribute_flag:
                    print("Rank: {}".format(rank), end=" ")
                print("Successfully trained the model for 40 batches!")
                print("--------------------")
                break

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
        train_duration_list = train_duration_list[1:]
        avg_batch_duration = sum(train_duration_list) / len(train_duration_list)

        with open(logs_path, "at") as logs_file:
            if distribute_flag:
                logs_file.write("Rank: {}, ".format(rank))
            logs_file.write(
                "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Epoch Duration: {:.3f} seconds\n".format(
                    epoch,
                    train_loss,
                    train_accuracy,
                    avg_batch_duration,
                    epoch_duration,
                )
            )
        if distribute_flag:
            print("Rank: {}".format(rank), end=" ")
        print(
            "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Epoch Duration: {:.3f} seconds".format(
                epoch, train_loss, train_accuracy, avg_batch_duration, epoch_duration
            )
        )
        print("--------------------")

        if distribute_flag:
            if rank == 0:
                ckpt_path = "{}/Rank_{}_Epoch_{}.pt".format(checkpoint_dir, rank, epoch)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_loss,
                    },
                    ckpt_path,
                )
        else:
            ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, epoch)
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

    if distribute_flag:
        dist.destroy_process_group()
    del avg_epoch_duration, epoch_duration_list
