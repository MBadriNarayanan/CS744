import os
import torch

from utils.general_utils import (
    create_helper_directories,
    generate_report,
    load_dataset,
    prepare_model_for_evaluation,
    prepare_model_for_training,
)
from utils.model import VGG11
from torch.nn.functional import log_softmax

torch.set_num_threads(4)


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

    generate_report(
        ground_truth=ground_truth,
        prediction=prediction,
        test_loss=test_loss,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )


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
