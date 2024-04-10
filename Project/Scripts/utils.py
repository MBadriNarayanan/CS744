import os
import time
import torch
import torch.cuda
import warnings

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
warnings.filterwarnings("ignore")


class GlueDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length,
        padding_value,
        truncation_flag,
        return_tensors,
        special_token_flag,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding_value = padding_value
        self.truncation_flag = truncation_flag
        self.return_tensors = return_tensors
        self.special_token_flag = special_token_flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_data = self.tokenizer(
            self.data[idx]["sentence"],
            max_length=self.max_length,
            padding=self.padding_value,
            truncation=self.truncation_flag,
            return_tensors=self.return_tensors,
            add_special_tokens=self.special_token_flag,
        )
        input_ids = torch.tensor(tokenized_data["input_ids"])
        attention_mask = torch.tensor(tokenized_data["attention_mask"])
        label = torch.tensor(self.data[idx]["label"])
        return (input_ids, attention_mask, label)


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
        report_path = os.path.join(logs_dir, "report.txt")
        print("Evaluation report will be stored at: {}!".format(report_path))
        return report_path


def generate_eval_report(
    ground_truth,
    prediction,
    test_loss,
    avg_batch_duration,
    test_duration,
    report_path,
    checkpoint_path,
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
            "Test Loss: {:.3f}, Test Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Test Duration: {:.3f} seconds\n".format(
                test_loss, test_accuracy, avg_batch_duration, test_duration
            )
        )
        report_file.write("Classification Report\n")
        report_file.write("{}\n".format(report))
        report_file.write("Confusion Matrix\n")
        report_file.write("{}\n".format(matrix))
        report_file.write("--------------------\n")

    del ground_truth, prediction
    del test_accuracy, report


def evaluate_model(
    model,
    device,
    test_loader,
    report_path,
    checkpoint_path,
):
    ground_truth = []
    prediction = []

    test_loss = 0.0
    test_duration = 0.0
    avg_batch_duration = 0.0

    test_start_time = time.time()

    model.eval()
    with torch.no_grad():
        for test_batch in tqdm(test_loader):
            input_ids = test_batch[0].view(input_ids.size(0), -1).to(device)
            attention_mask = test_batch[1].view(attention_mask.size(0), -1).to(device)
            labels = test_batch[2].view(labels.size(0), -1).to(device)

            batch_start_time = time.time()

            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            batch_loss = loss.item()

            y_pred = torch.argmax(logits, dim=1)
            y_pred = y_pred.cpu().tolist()
            target = labels.cpu().numpy().reshape(labels.shape[0]).tolist()

            test_loss += batch_loss
            ground_truth += target
            prediction += y_pred

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            avg_batch_duration += batch_duration

            del input_ids, attention_mask, labels
            del output, loss, logits
            del target, y_pred
            del batch_loss
            del batch_start_time, batch_end_time
            del batch_duration

    test_loss /= len(test_loader)
    avg_batch_duration /= len(test_loader)

    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    generate_eval_report(
        ground_truth=ground_truth,
        prediction=prediction,
        test_loss=test_loss,
        avg_batch_duration=avg_batch_duration,
        test_duration=test_duration,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )
    del avg_batch_duration
    del test_loss, test_duration
    del ground_truth, prediction


def get_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Parameters: {}".format(params))
    print("--------------------")


def generate_test_loader(
    test_data,
    tokenizer,
    max_length,
    padding_value,
    truncation_flag,
    return_tensors,
    special_token_flag,
    batch_size,
    shuffle_flag,
):
    test_data = GlueDataset(
        data=test_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle_flag)
    return test_loader


def generate_train_val_loader(
    train_data,
    val_data,
    tokenizer,
    max_length,
    padding_value,
    truncation_flag,
    return_tensors,
    special_token_flag,
    batch_size,
    shuffle_flag,
):
    train_data = GlueDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_flag)

    val_data = GlueDataset(
        data=val_data,
        tokenizer=tokenizer,
        max_length=max_length,
        padding_value=padding_value,
        truncation_flag=truncation_flag,
        return_tensors=return_tensors,
        special_token_flag=special_token_flag,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle_flag)

    return train_loader, val_loader


def prepare_base_model(model_name, label_count):
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=label_count
    )
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def prepare_model_for_evaluation(model, device, checkpoint_path):
    model = model.to(device)
    print("Loaded checkpoint:", checkpoint_path, "for evaluation!")
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
    optimizer,
    start_epoch,
    end_epoch,
    train_loader,
    val_loader,
    logs_path,
    checkpoint_dir,
):
    with open(logs_path, "at") as logs_file:
        logs_file.write(
            "Logs for the checkpoint stored at: {}/\n".format(checkpoint_dir)
        )

    number_of_epochs = end_epoch - start_epoch + 1
    total_steps = len(train_loader) * number_of_epochs
    training_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    avg_train_loss = 0.0
    avg_train_accuracy = 0.0
    avg_train_duration = 0.0
    avg_train_batch_time = 0.0

    avg_val_loss = 0.0
    avg_val_accuracy = 0.0
    avg_val_duration = 0.0
    avg_val_batch_time = 0.0

    for epoch in tqdm(range(start_epoch, end_epoch + 1)):
        model.train()

        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_train_duration = 0.0
        avg_train_batch_duration = 0.0

        train_epoch_start_time = time.time()

        for batch_idx, train_batch in enumerate(train_loader):
            input_ids = train_batch[0].view(input_ids.size(0), -1).to(device)
            attention_mask = train_batch[1].view(attention_mask.size(0), -1).to(device)
            labels = train_batch[2].view(labels.size(0), -1).to(device)

            batch_start_time = time.time()

            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits
            batch_loss = loss.item()
            clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
            training_scheduler.step()

            y_pred = torch.argmax(logits, dim=1)
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

        epoch_train_loss /= len(train_loader)
        epoch_train_accuracy /= len(train_loader)
        avg_train_batch_duration /= len(train_loader)

        train_epoch_end_time = time.time()
        epoch_train_duration = train_epoch_end_time - train_epoch_start_time

        avg_train_loss += epoch_train_loss
        avg_train_accuracy += epoch_train_accuracy
        avg_train_duration += epoch_train_duration
        avg_train_batch_time += avg_train_batch_duration

        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        epoch_val_duration = 0.0
        avg_val_batch_duration = 0.0

        val_epoch_start_time = time.time()

        model.eval()
        with torch.no_grad():
            for batch_idx, val_batch in enumerate(val_loader):
                input_ids = val_batch[0].view(input_ids.size(0), -1).to(device)
                attention_mask = (
                    val_batch[1].view(attention_mask.size(0), -1).to(device)
                )
                labels = val_batch[2].view(labels.size(0), -1).to(device)

                batch_start_time = time.time()

                output = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = output.loss
                logits = output.logits
                batch_loss = loss.item()

                y_pred = torch.argmax(logits, dim=1)
                y_pred = y_pred.cpu().numpy()
                target = labels.cpu().numpy().reshape(labels.shape[0])
                batch_accuracy = accuracy_score(target, y_pred)

                epoch_val_loss += batch_loss
                epoch_val_accuracy += batch_accuracy

                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                avg_val_batch_duration += batch_duration

                if batch_idx % 100 == 0:
                    write_string = "Epoch: {}, Val Batch Idx: {}, Val Batch Loss: {:.3f}, Val Batch Accuracy: {:.3f}, Val Batch Duration: {:.3f} seconds\n".format(
                        epoch, batch_idx, batch_loss, batch_accuracy, batch_duration
                    )
                    with open(logs_path, "at") as logs_file:
                        logs_file.write(write_string)
                    del write_string

                del input_ids, attention_mask, labels
                del output, loss, logits
                del target, y_pred
                del batch_loss, batch_accuracy
                del batch_start_time, batch_end_time
                del batch_duration

        epoch_val_loss /= len(val_loader)
        epoch_val_accuracy /= len(val_loader)
        avg_val_batch_duration /= len(val_loader)

        val_epoch_end_time = time.time()
        epoch_val_duration = val_epoch_end_time - val_epoch_start_time

        avg_val_loss += epoch_val_loss
        avg_val_accuracy += epoch_val_accuracy
        avg_val_duration += epoch_val_duration
        avg_val_batch_time += avg_val_batch_duration

        write_string = "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds, Val Loss: {:.3f}, Val Accuracy: {:.3f}, Val Duration: {:.3f} seconds, Avg Val Batch Duration: {:.3f} seconds\n".format(
            epoch,
            epoch_train_loss,
            epoch_train_accuracy,
            epoch_train_duration,
            avg_train_batch_duration,
            epoch_val_loss,
            epoch_val_accuracy,
            epoch_val_duration,
            avg_val_batch_duration,
        )
        with open(logs_path, "at") as logs_file:
            logs_file.write(write_string)
            logs_file.write("----------------------------------------------\n")
        del write_string

        ckpt_path = "{}/Epoch_{}.pt".format(checkpoint_dir, epoch)
        torch.save(
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

        del epoch_val_loss, epoch_val_accuracy
        del epoch_val_duration, avg_val_batch_duration
        del val_epoch_start_time, val_epoch_end_time

    avg_train_loss /= number_of_epochs
    avg_train_accuracy /= number_of_epochs
    avg_train_duration /= number_of_epochs
    avg_train_batch_time /= number_of_epochs
    avg_val_loss /= number_of_epochs
    avg_val_accuracy /= number_of_epochs
    avg_val_duration /= number_of_epochs
    avg_val_batch_time /= number_of_epochs

    write_string = "Avg Train Loss: {:.3f}, Avg Train Accuracy: {:.3f}, Avg Train Duration: {:.3f} seconds, Avg Train Batch Duration: {:.3f} seconds, Avg Val Loss: {:.3f}, Avg Val Accuracy: {:.3f}, Avg Val Duration: {:.3f} seconds, Avg Val Batch Duration: {:.3f} seconds,\n".format(
        avg_train_loss,
        avg_train_accuracy,
        avg_train_duration,
        avg_train_batch_time,
        avg_val_loss,
        avg_val_accuracy,
        avg_val_duration,
        avg_val_batch_time,
    )
    with open(logs_path, "at") as logs_file:
        logs_file.write(write_string)
        logs_file.write("----------------------------------------------\n")
    del write_string

    del avg_train_loss, avg_train_accuracy
    del avg_train_duration, avg_train_batch_time
    del avg_val_loss, avg_val_accuracy
    del avg_val_duration, avg_val_batch_time
