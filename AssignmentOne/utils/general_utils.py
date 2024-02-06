import os


def rename_csv_file(input_directory, output_csv_name="output.csv"):
    """
    Function to rename the CSV file to output csv name
    present in the input directory.

    I/P: Input Directory which contains a CSV file and
    the Output CSV name.
    O/P: Renames the CSV file to the given Output CSV name.
    """
    for file_name in os.listdir(input_directory):
        if file_name[0] != "." and file_name.endswith(".csv"):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_path = os.path.join(input_directory, output_csv_name)
            os.rename(input_file_path, output_file_path)
