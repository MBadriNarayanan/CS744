import argparse
import json

import torch

from accelerate import Accelerator
from datasets import load_dataset
from utils import (
    create_helper_directories,
    generate_data_loader,
    prepare_base_model,
    prepare_model_for_training,
    train_model,
)


def main(config):
    accelerator = Accelerator()
    num_gpus = accelerator.device_count
    print("Number of GPUs present: {}!".format(num_gpus))

    model_name = config["Model"]["modelName"]
    max_length = config["Model"]["sequenceLength"]
    padding_value = config["Model"]["paddingValue"]
    truncation_flag = config["Model"]["truncationFlag"]
    return_tensors = config["Model"]["returnTensors"]
    special_token_flag = config["Model"]["specialTokenFlag"]

    dataset_class = config["Dataset"]["datasetClass"]
    dataset_name = config["Dataset"]["datasetName"]
    label_count = config["Dataset"]["labelCount"]
    shuffle_flag = config["Dataset"]["shuffleFlag"]

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
    dataset = load_dataset(dataset_class, dataset_name)

    train_data = dataset["train"]
    train_loader = generate_data_loader(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
        batch_size=batch_size,
        shuffle_flag=shuffle_flag,
    )

    model, optimizer, training_scheduler = prepare_model_for_training(
        model=model,
        device=device,
        learning_rate=learning_rate,
        continue_flag=continue_flag,
        continue_checkpoint_path=continue_checkpoint_path,
    )

    model, optimizer, training_dataloader, training_scheduler = accelerator.prepare(
        model, optimizer, training_dataloader, training_scheduler
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
