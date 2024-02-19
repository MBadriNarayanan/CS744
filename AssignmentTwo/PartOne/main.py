import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from utils.general_utils import (
    create_helper_directories,
    evaluate_model,
    load_dataset,
    prepare_model_for_evaluation,
    prepare_model_for_training,
    train_model,
)
from utils.model import VGG11

torch.set_num_threads(4)
torch.manual_seed(42)


def main():
    print("\n--------------------\nStarting Part One!\n--------------------\n")

    batch_size = 256
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 0.0001
    start_epoch = 1
    end_epoch = 1
    continue_flag = False
    distribute_flag = False
    checkpoint_dir = "Checkpoints"
    logs_dir = "Logs"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU available!")
    else:
        device = torch.device("cpu")
        print("GPU not available!")

    model = VGG11()

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
        continue_checkpoint_path="",
        distribute_flag=distribute_flag,
        rank="",
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
        distribute_flag=distribute_flag,
    )

    checkpoint_path = os.path.join(checkpoint_dir, "Epoch_1.pt")

    model = prepare_model_for_evaluation(
        model=model, device=torch.device("cpu"), checkpoint_path=checkpoint_path
    )

    evaluate_model(
        model=model,
        device=torch.device("cpu"),
        criterion=criterion,
        test_loader=test_loader,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )

    print("\n--------------------\nPart One complete!\n--------------------\n")


if __name__ == "__main__":
    main()
