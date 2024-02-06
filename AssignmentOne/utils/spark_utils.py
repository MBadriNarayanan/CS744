from pyspark import SparkConf
from pyspark.sql import SparkSession


def create_spark_application(application_name):
    """
    Function to create a spark application
    with default config.

    I/P: Takes in the argument for the application name.
    O/P: Returns a spark application with the default config.
    """
    conf = SparkConf().setAppName(application_name).setMaster("local[2]")
    conf.set("spark.driver.memory", "30g")
    conf.set("spark.executor.memory", "30g")
    conf.set("spark.executor.cores", "5")
    conf.set("spark.task.cpus", "1")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark
