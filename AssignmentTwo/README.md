# Assignment Two

## Distributed Data Parallel Training: [Assignment Webiste](https://pages.cs.wisc.edu/~shivaram/cs744-sp24/assignment2.html)

## Environment Setup

The following commands were run across the 4 nodes to setup the environment

* `sudo apt update`
* `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
* Run the bash script and followed the installation instructions.
* `exec bash`
* `conda install numpy`
* `conda install scikit-learn`
* `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Code

```
def generate_eval_report(
    ground_truth, prediction, test_loss, report_path, checkpoint_path
):
    """
    This function make use of scikit-learn to generate the accuracy, classification
    report and confusion matrix and saves it in a report file.

    I/P:
        ground_truth: List of the ground truth values from the test set.
        prediction: List of prediction values made by the model.
        test_loss: Total Loss value calculated by the criterion for the entire test set.
        report_path: Path where the report file needs to be stored.
        checkpoint_path: Checkpoint with which the model was loaded to make the prediction.
    """
```

```
def evaluate_model(
    model,
    device,
    criterion,
    test_loader,
    report_path,
    checkpoint_path,
):
    ""'
    This function iterates over the test data set and gets the prediction. Based on the ground truth values and the prediction, CrossEntropyLoss is computed. After the loss value is calculated, log_softmax of the predictions is computed, which will be compared with the ground truth values to compute the accuracy and generate the evaluation report.

    This is because when CrossEntropyLoss is used, the torch definition computes the log_softmax before computing the loss.

    I/P:
        model: Model loaded with the weights from the checkpoint for which the prediction needs 
        device: Device to be used for making the predictions (cuda or cpu).
        criterion: Loss function to be used to evaluate the model.
        test_loader: The loader for the evaluation / test set.
        report_path:
        checkpoint_path: Checkpoint with which the model was loaded to make the prediction.
    """
```

```
def get_model_parameters(model):
    """
    This function takes in the model and tells us the number of trainable parameters.

    I/P:
        model: Model for which the number of trainable parameters is needed.
    """
```

```
def prepare_model_for_evaluation(
    model, 
    device, 
    checkpoint_path
):
    """
    This function prepares the model for evaluation.
    It uses the device passed as the argument and loads the checkpoint for which the model needs to be evaluated and returns the model.

    I/P:
        model: Model using which the evaluation needs to be made.
        device: Device to be used for making the predictions (cuda or cpu).
        checkpoint_path: Checkpoint with which the model needs to be loaded to make the prediction.
    """
```

```
def prepare_model_for_training(
    model,
    device,
    learning_rate,
    momentum,
    weight_decay,
    continue_flag,
    continue_checkpoint_path="",
    distribute_flag=False,
):
    """
    This function prepares the model for training.
    It also creates an instance of CrossEntropyLoss and moves the model and criterion to device.
    Given the learning_rate, momentum and weight_decay it creates an instance of SGD Optimizer, the optimizer for the model.
    If distribute flag is set as true, it creates a Distributed Data Parallel version of the model.
    If continue flag is set as true, continue checkpoint path needs to be given. The model and the optimizer is loaded with the weights and state dict of the checkpoint from which training needs to be continued.
    The number of trainable parameters in the model is also computed.

    The function returns the model, criterion and optimizer.

    I/P:
        model: Model using which the evaluation needs to be made.
        device: Device to be used for training (cuda or cpu).
        learning_rate: Learning rate for the SGD Optimizer.
        momentum: Momentum value for the SGD Optimizer.
        weight_decay: Weight Decay for the SGD Optimizer.
        continue_flag: Boolean flag to determine if the model needs to be trained from a previous checkpoint
        continue_checkpoint_path: Path for the checkpoint from which the training needs to continue.
        distribute_flag: Boolean value to determine if a Distributed Data parallel version of the model needs to be created.

    O/P:
        model, criterion and optimizer.
```

```
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
    distribute_flag,
    task_flag="",
    rank="",
)
    This function is used to train the model for (end_epoch - start_epoch + 1) iterations from start_epoch to end_epoch on the device specified. If distribute flag is given 
    Each line in the log file consists of the following 
    "Epoch: {}, Train Loss: {:.3f}, Train Accuracy: {:.3f}, Avg Batch Duration: {:.3f} seconds, Epoch Duration: {:.3f} seconds".
    
    If distribute flag a boolean value is set as True, it also logs the rank of the node on which it was trained.
    The task flag takes three value - "PartOne", "PartTwo" and "".
    If it is PartOne it trains the model using the constraints specified in Task Two Part One
    If it is PartTwo it trains the model using the constraints specified in Task Two Part Two

    If distribute flag is True or the task flag is PartOne or PartTwo, the trained model is returned
    
    For Part One
        distribute_flag = False
        task_flag = ""
    For Part Two
        Task One
            distribute_flag = False
            task_flag = PartOne
        Task One
            distribute_flag = False
            task_flag = PartTwo
    For Part Three
        distribute_flag = True
        task_flag = ""
```

## Part One: Training VGG-11 on CIFAR-10

* To train the model and get the performance, run the following command: `python3 PartOne/main.py`

## Part Two: Distributed Data Parallel Training

### Part Two Task One: Sync gradient with gather and scatter call using Gloo backend

* To train the model and get the performance, run the following command across the different: `python3 PartThree/main.py --master-ip $Master IP address$ --num-nodes $World Size / Number of nodes$ --rank $Rank / Current Node`

### Part Two Task Two: Sync gradient with allreduce using Gloo backend

* To train the model and get the performance, run the following command across the different: `python3 PartThree/main.py --master-ip $Master IP address$ --num-nodes $World Size / Number of nodes$ --rank $Rank / Current Node`

## Part Three: Distributed Data Parallel Training using Built in Module

* To train the model and get the performance, run the following command across the different: `python3 PartThree/main.py --master-ip $Master IP address$ --num-nodes $World Size / Number of nodes$ --rank $Rank / Current Node`