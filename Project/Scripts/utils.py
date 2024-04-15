import os
import time
import torch
import torch.cuda
import warnings

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.nn.functional import log_softmax
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, set_seed

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
set_seed(random_seed)
warnings.filterwarnings("ignore")


def create_directory(directory):
    try:
        os.mkdir(directory)
        print("Created directory: {}!".format(directory))
    except:
        print("Directory: {} already exists!".format(directory))


def create_helper_directories(checkpoint_dir, logs_dir, task_name, flag=True):
    create_directory(directory=checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, task_name)
    create_directory(directory=checkpoint_dir)

    create_directory(directory=logs_dir)
    logs_dir = os.path.join(logs_dir, task_name)
    create_directory(directory=logs_dir)

    if flag:
        logs_path = os.path.join(logs_dir, "logs.txt")
        print("Checkpoints will be stored at: {}!".format(checkpoint_dir))
        print("Training logs will be stored at: {}!".format(logs_path))
        return checkpoint_dir, logs_path
    else:
        validation_logs_dir = os.path.join(logs_dir, "Validation")
        create_directory(directory=validation_logs_dir)

        test_logs_dir = os.path.join(logs_dir, "Test")
        create_directory(directory=test_logs_dir)

        val_csv_path = os.path.join(validation_logs_dir, "output.csv")
        test_csv_path = os.path.join(test_logs_dir, "output.csv")

        val_report_path = os.path.join(validation_logs_dir, "report.txt")
        test_report_path = os.path.join(test_logs_dir, "report.txt")

        print("Validation reports will be stored at: {}!".format(validation_logs_dir))
        print("Test reports will be stored at: {}!".format(test_logs_dir))

        return val_csv_path, val_report_path, test_csv_path, test_report_path


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: {}".format(params))
    print("--------------------")


def get_data_loader(
    dataset_class, dataset_name, tokenizer, accelerator, batch_size, eval_flag=False
):
    def tokenize_function(example):
        outputs = tokenizer(example["sentence"], truncation=True, max_length=None)
        return outputs

    datasets = load_dataset(dataset_class, dataset_name)
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
        )

    def collate_fn(example):
        return tokenizer.pad(
            example,
            padding="longest",
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

    if eval_flag:
        val_dataloader = DataLoader(
            tokenized_datasets["validation"],
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )
        test_dataloader = DataLoader(
            tokenized_datasets["test"],
            shuffle=False,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=collate_fn,
        )
        return val_dataloader, test_dataloader

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=False,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_dataloader


def generate_report(
    ground_truth,
    prediction,
    loss,
    avg_batch_duration,
    duration,
    report_path,
    checkpoint_path,
):
    accuracy = accuracy_score(ground_truth, prediction)
    report = classification_report(
        ground_truth,
        prediction,
        digits=3,
        zero_division=0,
        target_names=["negative", "positive"],
    )
    matrix = confusion_matrix(ground_truth, prediction)

    with open(report_path, "w") as report_file:
        report_file.write(
            "Validation Metrics for the checkpoint: {}\n".format(checkpoint_path)
        )
        report_file.write(
            "Loss: {:.3f}, Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Duration: {:.3f} seconds\n".format(
                loss, accuracy, avg_batch_duration, duration
            )
        )
        report_file.write("Classification Report\n")
        report_file.write("{}\n".format(report))
        report_file.write("Confusion Matrix\n")
        report_file.write("{}\n".format(matrix))
        report_file.write("--------------------\n")

    del ground_truth, prediction
    del accuracy, report


def evaluate_model(
    model,
    device,
    data_loader,
    report_path,
    csv_path,
    checkpoint_path,
    flag="validation",
):
    sentence_data = []
    prediction = []

    evaluation_duration = 0.0
    avg_batch_duration = 0.0

    if flag == "validation":
        ground_truth = []
        val_loss = 0.0

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(data_loader):
            input_ids = data_batch["input_ids"]
            attention_mask = data_batch["attention_mask"]
            labels = data_batch["label"]
            sentence = data_batch["sentence"]

            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1).to(device)
            labels = labels.view(labels.size(0), -1).to(device)
            sentence = list(sentence)

            batch_start_time = time.time()

            if flag == "validation":
                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                logits = output.logits
                batch_loss = loss.item()

            else:
                output = model(input_ids, attention_mask=attention_mask)
                logits = output.logits

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().tolist()

            if flag == "validation":
                target = labels.cpu().numpy().reshape(labels.shape[0]).tolist()
                val_loss += batch_loss
                ground_truth += target

            prediction += y_pred
            sentence_data += sentence

            batch_end_time = time.time()
            avg_batch_duration += batch_end_time - batch_start_time

            del input_ids, attention_mask, labels
            del batch_start_time, batch_end_time
            del output, logits
            del y_pred, sentence

            if flag == "validation":
                del loss, target
                del batch_loss

    avg_batch_duration /= len(data_loader)

    end_time = time.time()
    evaluation_duration = end_time - start_time

    dataframe = pd.DataFrame()
    dataframe["Sentence"] = sentence_data
    dataframe["Prediction"] = prediction

    if flag == "validation":
        val_loss /= len(data_loader)
        generate_report(
            ground_truth=ground_truth,
            prediction=prediction,
            loss=val_loss,
            avg_batch_duration=avg_batch_duration,
            duration=evaluation_duration,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
        )
        dataframe["GroundTruth"] = ground_truth
        dataframe = dataframe[["Sentence", "GroundTruth", "Prediction"]]
        del val_loss
        del ground_truth

    dataframe.to_csv(csv_path, index=False)

    del avg_batch_duration
    del evaluation_duration
    del prediction, sentence_data
    del dataframe


def prepare_base_model(model_name, label_count):
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=label_count
    )
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_model_for_evaluation(model, device, checkpoint_path, flag="validation"):
    model = model.to(device)
    print("Loaded checkpoint: {} for {}!".format(checkpoint_path, flag))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("--------------------")
    return model


def prepare_model_for_training(
    model,
    device,
    learning_rate,
    continue_flag,
    continue_checkpoint_path="",
):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    if continue_flag:
        print("Model loaded for further training!")
        checkpoint = torch.load(continue_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Prepared model for training!")
    model.train()
    get_model_parameters(model=model)
    return model, optimizer


def train_model(
    model,
    device,
    accelerator,
    optimizer,
    start_epoch,
    end_epoch,
    data_loader,
    training_scheduler,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    number_of_epochs = end_epoch - start_epoch + 1

    avg_train_loss = 0.0
    avg_train_accuracy = 0.0
    avg_train_duration = 0.0
    avg_train_batch_time = 0.0

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_train_duration = 0.0
        avg_train_batch_duration = 0.0

        train_epoch_start_time = time.time()

        for batch_idx, data_batch in enumerate(data_loader):
            input_ids = data_batch["input_ids"]
            attention_mask = data_batch["attention_mask"]
            labels = data_batch["label"]

            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(attention_mask.size(0), -1).to(device)
            labels = labels.view(labels.size(0), -1).to(device)

            batch_start_time = time.time()

            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            batch_loss = loss.item()

            clip_grad_norm_(model.parameters(), 1.0)
            accelerator.backward(loss)
            optimizer.step()
            training_scheduler.step()

            y_pred = log_softmax(logits, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()
            target = labels.cpu().numpy().reshape(labels.shape[0])
            batch_accuracy = accuracy_score(target, y_pred)

            epoch_train_loss += batch_loss
            epoch_train_accuracy += batch_accuracy

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            avg_train_batch_duration += batch_duration

            if batch_idx % 100 == 0:
                write_string = "Epoch: {}, Train Batch Idx: {}, Train Batch Loss: {:.3f}, Train Batch Accuracy: {:.3f}, Train Batch Duration: {:.3f} seconds\n".format(
                    epoch, batch_idx, batch_loss, batch_accuracy, batch_duration
                )
                with open(logs_path, "at") as logs_file:
                    logs_file.write(write_string)
                del write_string

            torch.cuda.empty_cache()

            del input_ids, attention_mask, labels
            del output, loss, logits
            del target, y_pred
            del batch_loss, batch_accuracy
            del batch_start_time, batch_end_time
            del batch_duration

        epoch_train_loss /= len(data_loader)
        epoch_train_accuracy /= len(data_loader)
        avg_train_batch_duration /= len(data_loader)

        train_epoch_end_time = time.time()
        epoch_train_duration = train_epoch_end_time - train_epoch_start_time

        avg_train_loss += epoch_train_loss
        avg_train_accuracy += epoch_train_accuracy
        avg_train_duration += epoch_train_duration
        avg_train_batch_time += avg_train_batch_duration

        write_string = "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
            epoch,
            epoch_train_loss,
            epoch_train_accuracy,
            epoch_train_duration,
            avg_train_batch_duration,
        )
        with open(logs_path, "at") as logs_file:
            logs_file.write(write_string)
            logs_file.write("----------------------------------------------\n")
        del write_string

        ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, epoch)
        accelerator.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_train_loss,
            },
            ckpt_path,
        )

        del epoch_train_loss, epoch_train_accuracy
        del epoch_train_duration, avg_train_batch_duration
        del train_epoch_start_time, train_epoch_end_time

    avg_train_loss /= number_of_epochs
    avg_train_accuracy /= number_of_epochs
    avg_train_duration /= number_of_epochs
    avg_train_batch_time /= number_of_epochs

    write_string = "Avg Train Loss: {:.3f}, Avg Train Accuracy: {:.3f}, Avg Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds\n".format(
        avg_train_loss,
        avg_train_accuracy,
        avg_train_duration,
        avg_train_batch_time,
    )
    with open(logs_path, "at") as logs_file:
        logs_file.write(write_string)
        logs_file.write("----------------------------------------------\n")

    del write_string
    del avg_train_loss, avg_train_accuracy
    del avg_train_duration, avg_train_batch_time
