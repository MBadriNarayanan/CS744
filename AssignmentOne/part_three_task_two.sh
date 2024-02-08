#!/bin/bash

SPARK_HOME=/users/mbadnara/spark-3.3.4-bin-hadoop3
SPARK_MASTER_URL=spark://node0:7077

PYTHON_SCRIPT=/users/mbadnara/CS744/AssignmentOne/part_three.py
OUTPUT_DIRECTORY="hdfs://10.10.1.1:9000/user/mbadnara/data/output/TaskTwo"

SAMPLING_FLAG=1
PERSIST_FLAG=0

INPUT_PATH=hdfs://10.10.1.1:9000/user/mbadnara/data/web-BerkStan.txt
INPUT_PATH=/users/mbadnara/CS744/AssignmentOne/data/web-BerkStan.txt

INPUT_FLAG=1

EPOCHS=10
DAMPING_FACTOR=0.85

$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_PATH $DAMPING_FACTOR $EPOCHS $OUTPUT_DIRECTORY $INPUT_FLAG $SAMPLING_FLAG $PERSIST_FLAG

INPUT_FOLDER=hdfs://10.10.1.1:9000/user/mbadnara/data/enwiki-pages-articles
INPUT_FOLDER=/proj/uwmadison744-s24-PG0/data-part3/enwiki-pages-articles

INPUT_FLAG=0

$SPARK_HOME/bin/spark-submit --master $SPARK_MASTER_URL $PYTHON_SCRIPT $INPUT_FOLDER $DAMPING_FACTOR $EPOCHS $OUTPUT_DIRECTORY $SAMPLING_FLAG $PERSIST_FLAG