import os
from PIL import Image
import operator


def find_count(input_folder):
    counts = []
    for _, dirs, _ in os.walk(input_folder):
        for dir in dirs:
            subdirs = os.path.join(input_folder, dir)
            for _, subs, files in os.walk(subdirs):
                counts.append((dir, len(files)))
    counts.sort(key=operator.itemgetter(1))
    for c in counts:
        print c
    print len(counts)

if __name__ == '__main__':
    find_count("data/train")
