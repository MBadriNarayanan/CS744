import os
import sys

from pyspark import SparkContext

from utils.page_rank_utils import compute_page_rank, write_result

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Make sure all arguments are passed!")
        print("\n--------------------\nExited the application!\n--------------------\n")
        sys.exit()

    input_path = sys.argv[1]
    d = float(sys.argv[2])
    epochs = int(sys.argv[3])
    output_directory = sys.argv[4]
    input_flag = int(sys.argv[5])
    sampling_flag = int(sys.argv[6])
    persist_flag = int(sys.argv[7])

    if input_flag:
        print(
            "\n--------------------\nPerforming Page Rank algorithm for Berkley Graph Dataset!\n--------------------\n"
        )
    else:
        print(
            "\n--------------------\nPerforming Page Rank algorithm for Wiki Articles Dataset!\n--------------------\n"
        )

    sc = SparkContext("local", "PageRank")
    page_rank_result = compute_page_rank(
        sc=sc,
        file_path=input_path,
        d=d,
        epochs=epochs,
        flag=input_flag,
        sampling_flag=sampling_flag,
        persist_flag=persist_flag,
    )

    if input_flag:
        write_result(
            result=page_rank_result,
            output_path=output_directory,
            print_string="Berkley Graph Dataset",
        )
    else:
        write_result(
            result=page_rank_result,
            output_path=output_directory,
            print_string="Wiki Articles Dataset",
        )

    sc.stop()

    print(
        "\n--------------------\nPage Rank algorithm has been successfully!\n--------------------\n"
    )
