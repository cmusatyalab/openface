import numpy as np
import matplotlib.pyplot as plt

from mpltools import style
from mpltools import layout

style.use('ggplot')


def create_barchart(database=None, alg=None, loss=None,
                    accuracy=None, output=None):
    """
    ========
    Barchart
    ========

    A bar plot with errorbars and height labels on individual bars
    """

    N = len(accuracy[0])
    ind = np.arange(N)  # the x locations for the groups

    margin = 0.05
    width = (3. - margin) / N
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2)
    rects1 = ax.barh(ind + margin + (1 * width), accuracy[0], width, color=plt.rcParams['axes.color_cycle'][0])

    rects2 = ax.barh(ind + margin + (2 * width), accuracy[1], width, color=plt.rcParams['axes.color_cycle'][1])

    rects3 = ax.barh(ind + margin + (3 * width), accuracy[2], width, color=plt.rcParams['axes.color_cycle'][2])

    ax.set_ylabel('Training Time')
    ax.set_xlabel('Epoch Count')
    ax.set_title('Database: %s, Network: AlexNet, NN4, VGG, Train: GPU' % (database,))
    ax.set_yticks(ind + width)
    ax.set_yticklabels(loss)
    ax.legend([rects1, rects2, rects3], alg, loc='center right', ncol=1, fancybox=True, shadow=True)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            bl = rect.get_xy()
            x = rect.get_width() + bl[0] + 20
            y = bl[1]
            ax.text(x, y, "%.2f" % (rect.get_width()), ha='center', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.savefig(output, dpi=1200)


if __name__ == '__main__':
    database = "GaMO"
    # alg = "AlexNet"
    loss = ["TNet+Cross E", "Cross E.+C. Hinge", "C.Hinge", "Cross E.", "Cosine S.", "PHinge", "Triplet", "Distance R.",
            "KL D.", "Improved TNet", "Hadsell", "Double M.", "LMNN", "SoftPN", "Triplet G.", "LSSS"]
    # accuracy = [20.82, 6.56, 6.20, 6.42, 62.48, 60.49, 19.44, 19.22, 45.32, 19.32, 62.14, 61.80, 39.79, 19.75, 19.40,
    #             378.05]
    # output = "outputs/times/gamo_alexnet_gpu.png"
    # create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)
    #
    # alg = "NN4"
    # accuracy = [23.73, 10.85, 10.86, 10.82, 68.16, 64.96, 24.35, 24.56, 51.23, 24.44, 67.84, 66.23, 44.54, 24.28, 24.38,
    #             397.41]
    # output = "outputs/times/gamo_nn4_gpu.png"
    # create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)
    #
    # alg = "VGG"
    # accuracy = [39.75, 26.33, 26.36, 26.44, 81.62, 80.87, 39.19, 38.92, 65.56, 39.38, 81.36, 81.73, 59.78, 39.20, 39.38,
    #             403.05, ]
    # output = "outputs/times/gamo_vgg_gpu.png"
    # create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)
    accuracy = [[20.82, 6.56, 6.20, 6.42, 62.48, 60.49, 19.44, 19.22, 45.32, 19.32, 62.14, 61.80, 39.79, 19.75, 19.40,
                 378.05],
                [23.73, 10.85, 10.86, 10.82, 68.16, 64.96, 24.35, 24.56, 51.23, 24.44, 67.84, 66.23, 44.54, 24.28,
                 24.38, 397.41],
                [39.75, 26.33, 26.36, 26.44, 81.62, 80.87, 39.19, 38.92, 65.56, 39.38, 81.36, 81.73, 59.78, 39.20,
                 39.38, 403.05]]
    alg = ["AlexNet", "NN4", "VGG"]
    output = "outputs/times/gamo_gpu.png"
    create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)

    database = "CIFE"
    alg = ["AlexNet", "NN4", "VGG"]
    output = "outputs/times/cife_gpu.png"
    accuracy = [
        [22.19, 6.79, 6.88, 6.83, 69.68, 69.74, 21.83, 21.62, 52.20, 21.95, 69.41, 70.81, 45.13, 22.18, 21.88, 434.91],
        [26.46, 12.21, 12.62, 12.87, 76.37, 77.10, 28.44, 28.43, 61.17, 28.20, 77.41, 78.77, 53.64, 29.13, 29.55,
         470.74],
        [45.17, 30.37, 30.66, 30.40, 98.26, 97.57, 45.79, 45.77, 81.49, 46.00, 97.82, 98.11, 69.52, 45.78, 46.32,
         482.50]]
    create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)

    database = "MNIST"
    alg = ["AlexNet", "NN4", "VGG"]
    output = "outputs/times/mnist_gpu.png"
    accuracy = [
        [32.01, 18.30, 17.75, 17.77, 120.97, 120.64, 32.18, 30.46, 90.13, 30.30, 120.44, 120.76, 57.92, 31.61, 32.94,
         444.88],
        [30.85, 19.48, 19.58, 19.72, 127.56, 123.08, 31.65, 30.90, 96.62, 32.01, 125.81, 125.70, 60.44, 32.84, 32.58,
         468.36],
        [61.95, 50.63, 50.46, 50.59, 152.46, 148.37, 61.60, 61.33, 121.77, 61.93, 153.19, 151.05, 87.52, 62.16, 62.69,
         482.67]
    ]

    create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)

    database = "CIFAR-10"
    alg = ["AlexNet", "NN4", "VGG"]
    output = "outputs/times/cifar10_gpu.png"
    accuracy = [
        [

        ],
        [

        ],
        [

        ]
    ]

    create_barchart(database=database, alg=alg, loss=loss, accuracy=accuracy, output=output)
