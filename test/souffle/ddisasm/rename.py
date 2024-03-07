# python script to rename all .csv files to .facts in a directory

import os
import sys


def new_func(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            new_filename = os.path.splitext(filename)[0] + '.facts'
            os.rename(os.path.join(directory, filename),
                      os.path.join(directory, new_filename))


if __name__ == '__main__':
    directory = sys.argv[1]
    print(directory)
    new_func(directory)
