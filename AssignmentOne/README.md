# Assignment One

## Part One: Environment Setup

Using the instructions mentioned in this [website](https://pages.cs.wisc.edu/~shivaram/cs744-sp24/assignment1.html) the following tasks were completed

* Environment Setup in CloudLab machines.
* Installation of Apache Hadoop.
* Installation of Apache Spark.

## Part Two: Simple Spark Application

For this part, we were required to develop a simple Spark application to read a dataframe, sort the dataframe based on country code and timestamp values. The sorted dataframe needs to be saved in HDFS.
 
* Added the input file `export.csv` to the `data` folder.
* Added the helper function `create_spark_application()` to `spark_utils.py`. This function used to create a spark application based on the default config. It takes in application name as input parameter and returns the created spark application.
* The function `sort_data()` is added in the python script `part_two.py` which performs the task of sorting the dataframe based on the input columns. The input paramters are: Input Dataframe Path and Output Dataframe Folder.
* A bash script `part_two.sh` is created to run the spark application using the command: `$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_DATAFRAME_PATH $OUTPUT_DATAFRAME_FOLDER` with values for each variables set in the bash script.
* This script will create the sorted dataframe and saves the output CSV file at `$OUTPUT_DATAFRAME_FOLDER/output.csv`.
