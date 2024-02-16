import os
import torch
import torch.cuda

from tqdm import tqdm
import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.nn.functional import log_softmax
from torch.optim import SGD


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: {}".format(params))
    print("--------------------")


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
    train_dataset_length = len(train_loader.dataset)

    for epoch in range(start_epoch, end_epoch + 1):
        train_accuracy = 0.0
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
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

            batch_accuracy = accuracy_score(target, y_pred)
            train_loss += batch_loss
            train_accuracy += batch_accuracy

            if batch_idx % 20 == 0:
                print(
                    "Batch Idx: {}, Batch Loss: {:.3f}, Batch Accuracy: {:.3f}".format(
                        batch_idx, batch_loss, batch_accuracy
                    )
                )
                print("--------------------")

            torch.cuda.empty_cache()

            del data, target
            del y_hat, loss, y_pred
            del batch_loss, batch_accuracy

        train_loss = train_loss / train_dataset_length
        train_accuracy = train_accuracy / train_dataset_length

        with open(logs_path, "at") as logs_file:
            logs_file.write(
                "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}\n".format(
                    epoch, train_loss, train_accuracy
                )
            )
        print(
            "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}".format(
                epoch, train_loss, train_accuracy
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
        del train_loss, train_accuracy


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
        test_accuracy = 0.0
        test_loss = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):
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
