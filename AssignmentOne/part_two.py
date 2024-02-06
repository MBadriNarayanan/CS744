import sys

from pyspark.sql.functions import col

from utils.general_utils import rename_csv_file
from utils.spark_utils import create_spark_application


def sort_data(input_dataframe_path, output_dataframe_folder):
    """
    Function to sort the dataframe based
    on country_code and time stamp.

    I/P: Input Dataframe Path and Output Dataframe Folder
    O/P: Sorts the dataframe based on given criteria and
    renames it to output.csv.
    """
    spark = create_spark_application(application_name="SortData")
    df = spark.read.csv(input_dataframe_path, header=True)
    df = df.orderBy(col("cca2"), col("timestamp")).repartition(1)
    df.write.csv(output_dataframe_folder, header=True)
    spark.stop()
    rename_csv_file(
        input_directory=output_dataframe_folder, output_csv_name="output.csv"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Make sure Input Dataframe path and Output Folder are passed as argument!"
        )
        print("\n--------------------\nExited the application!\n--------------------\n")
        sys.exit()

    input_dataframe_path = sys.argv[1]
    output_dataframe_folder = sys.argv[2]

    sort_data(
        input_dataframe_path=input_dataframe_path,
        output_dataframe_folder=output_dataframe_folder,
    )
    print(
        "\n--------------------\nData has been sorted successfully!\n--------------------\n"
    )
