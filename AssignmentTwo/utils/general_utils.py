import os
import torch
import torch.cuda

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
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


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def create_helper_directories(checkpoint_dir, logs_dir, task_name, flag=False):
    create_directory(directory=checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, "PartOne")
    create_directory(directory=checkpoint_dir)

    create_directory(directory=logs_dir)
    logs_dir = os.path.join(logs_dir, "PartOne")
    create_directory(directory=logs_dir)

    if flag:
        checkpoint_dir = os.path.join(checkpoint_dir, task_name)
        logs_dir = os.path.join(logs_dir, task_name)
        create_directory(directory=checkpoint_dir)
        create_directory(directory=logs_dir)

    logs_path = os.path.join(logs_dir, "logs.txt")
    report_path = os.path.join(logs_dir, "report.txt")

    print("Checkpoints will be stored at: {}!".format(checkpoint_dir))
    print("Training logs will be stored at: {}!".format(logs_path))
    print("Evaluation report will be stored at: {}!".format(report_path))

    return checkpoint_dir, logs_path, report_path


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
):
    model = model.to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

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
