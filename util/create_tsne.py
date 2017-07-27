import os


def create_tsne(train, path, max_num):
    print path, str(max_num)
    if not train:
        if 'cife' in path or 'gamo' in path:
            os.system(
                "python tsne.py %s/rep-%s/test --names Angry Disgust Fear Happy Neutral Sad Surprise" % (
                    path, str(max_num)))
        elif "cifar" in path:
            os.system(
                "python tsne.py %s/rep-%s/test --names Airplane Automobile Bird Cat Deer Dog Frog Horse Ship Truck" % (
                    path, str(max_num)))
        elif 'mnist' in path:
            os.system(
                "python tsne.py %s/rep-%s/test --names 0 1 2 3 4 5 6 7 8 9" % (
                    path, str(max_num)))
    else:
        if 'cife' in path or 'gamo' in path:
            os.system(
                "python tsne.py %s/rep-%s/train --names Angry Disgust Fear Happy Neutral Sad Surprise" % (
                    path, str(max_num)))
        elif 'cifar10' in path:
            os.system(
                "python tsne.py %s/rep-%s/train --names Airplane Automobile Bird Cat Deer Dog Frog Horse Ship Truck" % (
                    path, str(max_num)))
        elif 'mnist' in path:
            os.system(
                "python tsne.py %s/rep-%s/train --names 0 1 2 3 4 5 6 7 8 9" % (
                    path, str(max_num)))
