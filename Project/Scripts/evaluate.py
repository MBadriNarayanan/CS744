import argparse
import json

import torch

from datasets import load_dataset
from utils import (
    create_helper_directories,
    get_data_loader,
    prepare_base_model,
    prepare_model_for_evaluation,
    evaluate_model,
)


def main(config):
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

    batch_size = config["Eval"]["batchSize"]
    checkpoint_path = config["Eval"]["checkpointPath"]

    (
        val_csv_path,
        val_report_path,
        test_csv_path,
        test_report_path,
    ) = create_helper_directories(
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        task_name=task_name,
        flag=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
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

    val_data = dataset["validation"]
    val_loader = get_data_loader(
        data=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
        batch_size=batch_size,
        shuffle_flag=shuffle_flag,
    )

    test_data = dataset["test"]
    test_loader = get_data_loader(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
        batch_size=batch_size,
        shuffle_flag=shuffle_flag,
    )

    model = prepare_model_for_evaluation(
        model=model, device=device, checkpoint_path=checkpoint_path, flag="validation"
    )
    evaluate_model(
        model=model,
        device=device,
        data_loader=val_loader,
        report_path=val_report_path,
        csv_path=val_csv_path,
        checkpoint_path=checkpoint_path,
        flag="validation",
    )

    model = prepare_model_for_evaluation(
        model=model, device=device, checkpoint_path=checkpoint_path, flag="testing"
    )
    evaluate_model(
        model=model,
        device=device,
        data_loader=test_loader,
        report_path=test_report_path,
        csv_path=test_csv_path,
        checkpoint_path=checkpoint_path,
        flag="testing",
    )


if __name__ == "__main__":
    print("\n--------------------\nStarting model evaluation!\n--------------------\n")

    parser = argparse.ArgumentParser(description="Argparse for Model evaluation")
    parser.add_argument(
        "--config",
        "-C",
        type=str,
        help="Config file for model evaluation",
        required=True,
    )
    args = parser.parse_args()

    json_filename = args.config
    with open(json_filename, "rt") as json_file:
        config = json.load(json_file)

    main(config=config)
    print("\n--------------------\nModel evaluation completed!\n--------------------\n")
