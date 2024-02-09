import random
import re
import sys

from pyspark.sql import SparkSession


def get_edges(line):
    nodes = re.split(r"\s+", line, maxsplit=2)
    return nodes[0], nodes[1]


def compute_contributions(urls, rank):
    n = len(urls)
    contributions = []
    for url in urls:
        contributions.append((url, rank / n))
    return contributions


def custom_partition(data, num_partitions):
    data = data.collect()
    partition_size = len(data) // num_partitions
    random.shuffle(data)

    partitions = []
    for idx in range(num_partitions):
        partitions.append(data[idx * partition_size : (idx + 1) * partition_size])

    remainder = len(data) % num_partitions
    for idx in range(remainder):
        partitions[idx].append(data[num_partitions * partition_size + idx])
    return partitions


def compute_page_rank(
    spark, input_path, epochs, d, sampling_flag=True, persist_flag=True
):
    df = spark.sparkContext.textFile(input_path)
    input_lines = df.filter(lambda line: not line.startswith("#"))
    nodes = input_lines.map(get_edges)
    if sampling_flag:
        partitions = custom_partition(nodes, num_partitions=20)
        data = []
        for partition in partitions:
            data.append(partition)
        nodes = data[0]
    ranks = nodes.groupByKey().mapValues(lambda data: 1.0)

    for epoch in range(epochs):
        contributions = nodes.join(ranks).flatMap(
            lambda neighbor_page_rank: compute_contributions(
                neighbor_page_rank[1][0], neighbor_page_rank[1][1]
            )
        )
        page_ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(
            lambda rank: (1 - d) + d * rank
        )
        if persist_flag:
            page_ranks.persist()
        print(
            "\n--------------------\n[ {} / {} ] Completed!\n--------------------\n".format(
                epoch + 1, epochs
            )
        )
    return page_ranks


def write_result(page_rank_result, output_path):
    hdfs_text_file_path = "{}/{}".format(output_path, "wiki.txt")
    page_rank_result.saveAsTextFile(hdfs_text_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Make sure all arguments are passed!")
        print("\n--------------------\nExited the application!\n--------------------\n")
        sys.exit()

    input_path = sys.argv[1]
    d = float(sys.argv[2])
    epochs = int(sys.argv[3])
    output_directory = sys.argv[4]
    sampling_flag = int(sys.argv[5])
    persist_flag = int(sys.argv[6])

    print(
        "\n--------------------\nPerforming Page Rank algorithm for Wiki Articles Dataset!\n--------------------\n"
    )

    spark = (
        SparkSession.builder.appName("PageRank")
        .master("spark://c220g5-120130vm-1.wisc.cloudlab.us")
        .getOrCreate()
    )

    if input_path[-1] == "/":
        input_path = input_path + "*"
    else:
        input_path = "/*"

    page_rank_result = compute_page_rank(
        spark=spark,
        input_path=input_path,
        d=d,
        epochs=epochs,
        sampling_flag=sampling_flag,
        persist_flag=persist_flag,
    )
    write_result(
        page_rank_result=page_rank_result,
        output_path=output_directory,
    )

    print(
        "\n--------------------\nPage Rank algorithm has been successfully completed!\n--------------------\n"
    )
