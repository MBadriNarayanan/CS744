import argparse
import json

import torch

from accelerate import Accelerator
from utils import (
    create_helper_directories,
    get_data_loader,
    prepare_base_model,
    prepare_model_for_evaluation,
    evaluate_model,
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

    val_loader = get_data_loader(
        dataset_class=dataset_class,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=batch_size,
        eval_flag=False,
    )
    test_loader = get_data_loader(
        dataset_class=dataset_class,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        accelerator=accelerator,
        batch_size=batch_size,
        eval_flag=False,
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
