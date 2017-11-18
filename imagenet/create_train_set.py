import os
from PIL import Image


def get_train_set(input_folder, output_folder, dim):
    for _, dirs, _ in os.walk(input_folder):
        for dir in dirs:
            subdirs = os.path.join(input_folder, dir)
            for _, subs, _ in os.walk(subdirs):
                for sub in subs:
                    sub_subdirs = os.path.join(subdirs, sub)
                    for _, ss, files in os.walk(sub_subdirs):
                        if ss:
                            for s in ss:
                                sub_sub_subdirs = os.path.join(sub_subdirs, s)
                                for _, _, fs in os.walk(sub_sub_subdirs):
                                    for ff in fs:
                                        try:
                                            output_folder_name = os.path.join(output_folder, sub)
                                            if not os.path.exists(output_folder_name):
                                                os.makedirs(output_folder_name)
                                            out_filepath = os.path.join(output_folder_name, '%s_%s' % (sub, ff))
                                            in_filepath = os.path.join(sub_sub_subdirs, ff)
                                            im = Image.open(in_filepath)
                                            im = im.resize((dim, dim))
                                            im.save(out_filepath)
                                        except Exception as e:
                                            print e.message, in_filepath
                        else:
                            for file in files:
                                try:
                                    output_folder_name = os.path.join(output_folder, sub)
                                    if not os.path.exists(output_folder_name):
                                        os.makedirs(output_folder_name)
                                    out_filepath = os.path.join(output_folder_name, file)
                                    in_filepath = os.path.join(sub_subdirs, file)
                                    im = Image.open(in_filepath)
                                    im = im.resize((dim, dim))
                                    im.save(out_filepath)
                                except Exception as e:
                                    print e.message, in_filepath


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--outputDir', type=str, help='an integer for the accumulator')
    parser.add_argument('--dim', type=int, default=64)

    args = parser.parse_args()
    get_train_set(args.inputDir, args.outputDir, args.dim)
