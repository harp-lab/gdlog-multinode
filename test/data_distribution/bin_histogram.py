import csv
import argparse
import sys
import os
import matplotlib.pyplot as plt


def parse_command_line_args():
    parser = argparse.ArgumentParser(prog="Data Distribution Bin Histogram",
                                     description="Show the data distribution of a relation",
                                     epilog="Uses the SLOG's hashing method")
    parser.add_argument('-d', '--data', nargs="?", default=None, help="Path to the dataset (optional)")
    parser.add_argument('-np', '--ranks', nargs="?", type=int, default=None, help="Total ranks (optional)")
    parser.add_argument('-o', '--output', nargs="?", default=None, help="Output directory (optional)")
    args = parser.parse_args()
    return args.data, args.ranks, args.output


def read_csv(filename, delimiter='\t', header=None, row_size=None):
    data = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        if header is not None:
            next(csv_reader)
        for row in csv_reader:
            data.append(list(map(int, row)))
            if row_size is not None and len(data) >= row_size:
                break
    return data


def get_row_size(data_path):
    with open(data_path, 'r') as f:
        row_size = 0
        for line in f:
            if line.endswith('\n'):
                row_size += 1
    return row_size


def get_dataset(data_path):
    try:
        row_size = get_row_size(data_path)
        delimiter = '\t'
        header = None
        result = read_csv(data_path, delimiter=delimiter, header=header, row_size=row_size)
        return result
    except Exception as ex:
        print(f"Error in getting dataset: {ex}")
        print("Exiting the program.")
        sys.exit()


def get_prefix_hash(row_data, prefix_length):
    base = 2166136261
    prime = 16777619
    hash_value = base
    for i in range(prefix_length):
        chunk = row_data[i]
        hash_value ^= chunk & 255
        hash_value *= prime
        for j in range(3):
            chunk >>= 8
            hash_value ^= chunk & 255
            hash_value *= prime
    return hash_value


def get_histogram(dataset, total_rank):
    bins = [0 for i in range(total_rank)]
    for row in dataset:
        hash_value = get_prefix_hash(row, 1)
        calculated_rank = hash_value % total_rank
        bins[calculated_rank] += 1
    return bins


def draw_histogram(dataset_name, data_counts, output_dir):
    total_ranks = len(data_counts)
    ranks = range(1, len(data_counts) + 1)
    # Set the figure size
    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(24, 8)
    # Create a histogram
    bars = plt.bar(ranks, data_counts, width=0.7)
    # Add labels and title
    plt.xlabel(f'Rank')
    plt.ylabel('Count')
    plt.title(f'Data Distribution for {dataset_name} using {total_ranks} ranks')
    # Set integer ticks on the x-axis
    plt.xticks(list(map(int, plt.xticks()[0])))
    # Set integer ticks on the y-axis
    plt.yticks(list(map(int, plt.yticks()[0])))
    # Set x-axis limits to start at 0
    plt.xlim(0, max(ranks) + 1)
    # Add value annotations above each bar
    # for bar in bars:
    #     y_val = bar.get_height()
    #     offset = 0.01 * max(data_counts)
    #     plt.text(bar.get_x() + bar.get_width() / 2, y_val + offset, round(y_val, 1), ha='center', va='bottom',
    #              rotation='vertical')

    image_path = os.path.join(output_dir, f'{dataset_name}_{total_ranks}.png')
    image_path = image_path.replace(" ", "")
    # This should be called after all axes have been added
    fig.tight_layout()
    # plt.show()
    plt.savefig(image_path, bbox_inches='tight', dpi=400)
    plt.close()
    print(f"Generated {image_path}")


def main():
    total_rank = 256
    dataset_path, total_rank, output_dir = parse_command_line_args()
    if output_dir == None:
        output_dir = str(total_rank)
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "ego-Facebook": "../../data/data_88234.txt",
        "wiki-Vote": "../../data/data_103689.txt",
        "luxembourg_osm": "../../data/data_119666.txt",
        "fe_sphere": "../../data/data_49152.txt",
        "fe_body": "../../data/data_163734.txt",
        "cti": "../../data/data_48232.txt",
        "fe_ocean": "../../data/data_409593.txt",
        "wing": "../../data/data_121544.txt",
        "loc-Brightkite": "../../data/data_214078.txt",
        "delaunay_n16": "../../data/data_196575.txt",
        "usroads": "../../data/data_165435.txt",
        "CA-HepTh": "../../data/data_51971.txt",
        "SF.cedge": "../../data/data_223001.txt",
        "p2p-Gnutella31": "../../data/data_147892.txt",
        "p2p-Gnutella09": "../../data/data_26013.txt",
        "p2p-Gnutella04": "../../data/data_39994.txt",
        "cal.cedge": "../../data/data_21693.txt",
        "TG.cedge": "../../data/data_23874.txt",
        "OL.cedge": "../../data/data_7035.txt",
        # "data 10": "../../data/data_10.txt",
    }
    if not dataset_path:
        for dataset_name, dataset_path in datasets.items():
            dataset = get_dataset(dataset_path)
            data_counts = get_histogram(dataset, total_rank)
            draw_histogram(dataset_name, data_counts, output_dir)
    else:
        dataset = get_dataset(dataset_path)
        data_counts = get_histogram(dataset, total_rank)
        dataset_name = dataset_path.split("/")[-1]
        draw_histogram(dataset_name, data_counts, output_dir)


if __name__ == "__main__":
    main()

# Running example:
# python bin_histogram.py -np 256
# python bin_histogram.py -np 512
# python bin_histogram.py -d ../../data/data_10.txt -np 3 -o sample_data
# python bin_histogram.py -h
# usage: Data Distribution Bin Histogram [-h] [-d [DATA]] [-np [RANKS]] [-o [OUTPUT]]