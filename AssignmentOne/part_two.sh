#!/bin/bash

SPARK_HOME=/users/mbadnara/spark-3.3.4-bin-hadoop3
SPARK_MASTER_URL=spark://node0:7077

PYTHON_SCRIPT=/users/mbadnara/CS744/AssignmentOne/part_two.py
INPUT_DATAFRAME_PATH=/users/mbadnara/CS744/AssignmentOne/data/export.csv
OUTPUT_DATAFRAME_FOLDER=/users/mbadnara/CS744/AssignmentOne/data/output

$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_DATAFRAME_PATH $OUTPUT_DATAFRAME_FOLDER
