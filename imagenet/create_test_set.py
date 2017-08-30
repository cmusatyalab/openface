import os
from PIL import Image


def get_test_set(mapping_path, test_mapping, test_folder, output, dim):
    with open(mapping_path, mode='rb') as f:
        train_data = f.readlines()
    train_classes = {}
    test_classes = {}
    for d in train_data:
        path, cls = d.replace('\n', '').split(" ")
        cls = int(cls)
        splitted = path.split('/')
        clsname = splitted[2]
        train_classes[cls] = clsname
    with open(test_mapping, mode='rb') as f:
        test_data = f.readlines()
    for d in test_data:
        path, cls = d.replace('\n', '').split(" ")
        cls = int(cls)
        if cls in train_classes.keys():
            if not cls in test_classes:
                test_classes[cls] = [path]
            else:
                test_classes[cls].append(path)

    for i, im_names in test_classes.iteritems():
        directory = os.path.join(output, train_classes[i])
        if not os.path.exists(directory):
            os.makedirs(directory)
        for im_name in im_names:
            print directory
            out_filepath = os.path.join(directory, im_name)
            in_filepath = os.path.join(test_folder, im_name)
            im = Image.open(in_filepath)
            im = im.resize((dim, dim))
            im.save(out_filepath)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trainMapping', type=str, help='an integer for the accumulator')
    parser.add_argument('--testMapping', type=str, help='an integer for the accumulator')
    parser.add_argument('--testFolder', type=str, help='an integer for the accumulator')
    parser.add_argument('--outputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--dim', type=int, default=64)

    args = parser.parse_args()
    get_test_set(args.trainMapping, args.testMapping, args.testFolder, args.outputDir, args.dim)
