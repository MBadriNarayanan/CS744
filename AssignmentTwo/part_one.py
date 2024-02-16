import os
import torch

from model import VGG11
from utils import (
    create_directory,
    evaluate_model,
    prepare_model_for_evaluation,
    prepare_model_for_training,
    train_model,
)

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

torch.set_num_threads(4)


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

    model = VGG11()
    model.to(device)

    create_directory(directory=checkpoint_dir)
    create_directory(directory=logs_dir)

    logs_path = os.path.join(logs_dir, "logs_part_one.txt")
    report_path = os.path.join(logs_dir, "report_part_one.txt")

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
