import argparse
import json

import torch

from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
from utils import (
    create_helper_directories,
    get_data_loader,
    prepare_base_model,
    prepare_model_for_training,
    train_model,
)


def main(config):
    accelerator = Accelerator()
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs present: {}!".format(num_gpus))

    model_name = config["Model"]["modelName"]

    dataset_class = config["Dataset"]["datasetClass"]
    dataset_name = config["Dataset"]["datasetName"]
    label_count = config["Dataset"]["labelCount"]

    checkpoint_dir = config["Logs"]["checkpointDirectory"]
    logs_dir = config["Logs"]["logsDirectory"]
    task_name = config["Logs"]["taskName"]

    batch_size = config["Train"]["batchSize"]
    start_epoch = config["Train"]["startEpoch"]
    end_epoch = config["Train"]["endEpoch"]
    learning_rate = config["Train"]["learningRate"]
    continue_flag = config["Train"]["continueFlag"]
    continue_checkpoint_path = config["Train"]["continueCheckpoint"]

    checkpoint_dir, logs_path = create_helper_directories(
        checkpoint_dir=checkpoint_dir, logs_dir=logs_dir, task_name=task_name, flag=True
    )

    if torch.cuda.is_available():
        device = accelerator.device
        print("CUDA available!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Metal available!")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU!")

    torch.cuda.empty_cache()

    model, tokenizer = prepare_base_model(
        model_name=model_name, label_count=label_count
    )
    train_loader = get_data_loader(
        dataset_class=dataset_class,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=batch_size,
        eval_flag=False,
    )

    model, optimizer = prepare_model_for_training(
        model=model,
        device=device,
        learning_rate=learning_rate,
        continue_flag=continue_flag,
        continue_checkpoint_path=continue_checkpoint_path,
    )

    num_epochs = end_epoch - start_epoch + 1
    training_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    model, optimizer, train_loader, training_scheduler = accelerator.prepare(
        model, optimizer, train_loader, training_scheduler
    )

    train_model(
        model=model,
        device=device,
        accelerator=accelerator,
        optimizer=optimizer,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        data_loader=train_loader,
        training_scheduler=training_scheduler,
        logs_path=logs_path,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    print("\n--------------------\nStarting model training!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model training")
    parser.add_argument(
        "--config", "-C", type=str, help="Config file for model training", required=True
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel training completed!\n--------------------\n")
