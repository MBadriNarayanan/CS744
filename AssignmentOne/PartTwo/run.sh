#!/bin/bash

SPARK_HOME=/users/mbadnara/spark-3.3.4-bin-hadoop3
SPARK_MASTER_URL=spark://node0:7077

PYTHON_SCRIPT=/users/mbadnara/CS744/AssignmentOne/PartTwo/script.py
INPUT_DATAFRAME_PATH=hdfs://10.10.1.1:9000/user/mbadnara/data/export.csv
OUTPUT_DATAFRAME_PATH=hdfs://10.10.1.1:9000/user/mbadnara/data/output/output.csv

$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_DATAFRAME_PATH $OUTPUT_DATAFRAME_PATH
