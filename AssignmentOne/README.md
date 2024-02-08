# Assignment One

## Part One: Environment Setup

Using the instructions mentioned in this [website](https://pages.cs.wisc.edu/~shivaram/cs744-sp24/assignment1.html) the following tasks were completed

* Environment Setup in CloudLab machines.
* Installation of Apache Hadoop.
* Installation of Apache Spark.

## Part Two: Simple Spark Application

For this part, we were required to develop a simple Spark application to read a dataframe, sort the dataframe based on country code and timestamp values. The sorted dataframe needs to be saved in HDFS.
 
* Added the input file `export.csv` to the `data` folder in HDFS using this command `hadoop fs -put /users/mbadnara/CS744/AssignmentOne/data/export.csv hdfs://10.10.1.1:9000/user/mbadnara/data/`.
* Added the helper function `create_spark_application()` to `spark_utils.py`. This function used to create a spark application based on the default config. It takes in application name as input parameter and returns the created spark application.
* The function `sort_data()` is added in the python script `part_two.py` which performs the task of sorting the dataframe based on the input columns. The input paramters are: Input Dataframe Path and Output Dataframe Path.
* A bash script `part_two.sh` is created to run the spark application using the command: `$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_DATAFRAME_PATH $OUTPUT_DATAFRAME_PATH` with values for each variables set in the bash script.
* This script will create the sorted dataframe and saves the output CSV file at `$OUTPUT_DATAFRAME_PATH` in HDFS.

## Part Three: Page Rank

For this part, we were required to implement the PageRank algorithm.

### Task One

#### Write a Scala/Python/Java Spark application that implements the PageRank algorithm.

### Task Two

#### In order to achieve high parallelism, Spark will split the data into smaller chunks called partitions which are distributed across different nodes in the cluster. Partitions can be changed in several ways. For example, any shuffle operation on a DataFrame (e.g., join()) will result in a change in partitions (customizable via user’s configuration). In addition, one can also decide how to partition data when writing DataFrames back to disk. For this task, add appropriate custom DataFrame/RDD partitioning and see what changes.

### Task Three

#### Persist the appropriate DataFrame/RDD(s) as in-memory objects and see what changes.

### Task Four

#### Kill a Worker process and see the changes. You should trigger the failure to a desired worker VM when the application reaches 25% and 75% of its lifetime.