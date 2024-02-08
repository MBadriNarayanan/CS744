#!/bin/bash

SPARK_HOME=/users/mbadnara/spark-3.3.4-bin-hadoop3
HADOOP_HOME=/users/mbadnara/hadoop-3.3.6/sbin

$HADOOP_HOME/stop-dfs.sh
$SPARK_HOME/sbin/stop-all.sh

$HADOOP_HOME/start-dfs.sh
$SPARK_HOME/sbin/start-all.sh