import os
import random

from pyspark.storagelevel import StorageLevel


def parse_neighbors(line, flag):
    line = line.split("\t")
    if flag:
        return int(line[0]), int(line[1])
    else:
        return line[0], line[1]


def compute_contribs(neighbor_page_rank):
    data = neighbor_page_rank[1][0]
    rank = neighbor_page_rank[1][1]
    for neighbor in data:
        yield neighbor, rank / len(data)


def page_rank(links, epochs, d, persist_flag):
    ranks = links.mapValues(lambda neighbors: 1.0)
    for epoch in range(epochs):
        contributions = links.join(ranks).flatMap(
            lambda neighbor_page_rank: compute_contribs(neighbor_page_rank)
        )
        ranks = contributions.reduceByKey(lambda x, y: x + y).mapValues(
            lambda rank: (1 - d) + d * rank
        )
        if persist_flag:
            ranks.persist()
    return ranks


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
    sc, file_path, epochs, d, flag=True, sampling_flag=True, persist_flag=True
):
    if flag:
        input_lines = sc.textFile(file_path)
        input_lines = input_lines.filter(lambda line: not line.startswith("#"))
        input_links = input_lines.map(lambda line: parse_neighbors(line, flag=True))
    else:
        input_links = []
        for input_file in os.listdir(file_path):
            if input_file[0] != "." and input_file.startswith("link"):
                input_file_path = os.path.join(file_path, input_file)
                input_lines = sc.textFile(input_file_path)
                input_link = input_lines.map(
                    lambda line: parse_neighbors(line, flag=False)
                )
                input_links.append(input_link)
        input_links = sc.union(input_links)
    if sampling_flag:
        partitions = custom_partition(input_links, num_partitions=20)
        data = []
        for partition in partitions:
            data.append(sc.parallelize(partition))
        input_links = data[0]
    input_links = input_links.distinct().groupByKey()
    page_ranks = page_rank(
        links=input_links, epochs=epochs, d=d, persist_flag=persist_flag
    )
    result = page_ranks.collect()
    result = sc.parallelize(result)
    return result


def write_result(page_rank_result, output_path, print_string):
    if print_string == "Berkley Graph Dataset":
        hdfs_text_file_path = "{}/{}".format(output_path, "graph.txt")
    else:
        hdfs_text_file_path = "{}/{}".format(output_path, "wiki.txt")
    page_rank_result.saveAsTextFile(hdfs_text_file_path)
