#!/bin/bash

SPARK_HOME=/users/mbadnara/spark-3.3.4-bin-hadoop3
SPARK_MASTER_URL=spark://node0:7077

PYTHON_SCRIPT=/users/mbadnara/CS744/AssignmentOne/PartThree/script.py
INPUT_PATH="hdfs://10.10.1.1:9000/user/mbadnara/data/enwiki-pages-articles/"
DAMPING_FACTOR=0.85
EPOCHS=10
OUTPUT_DIRECTORY="hdfs://10.10.1.1:9000/user/mbadnara/data/output/TaskOne"
SAMPLING_FLAG=0
PERSIST_FLAG=0

$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_PATH $DAMPING_FACTOR $EPOCHS $OUTPUT_DIRECTORY $SAMPLING_FLAG $PERSIST_FLAG