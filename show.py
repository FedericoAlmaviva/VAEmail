import random
import re
import matplotlib.pyplot as plt


def parseLogEpochs(filepth):
    ret = list()
    with open(filepth, 'r') as file_object:
        line = file_object.readline()
        while (line):
            ld = re.findall('\d+\.\d+|\d+', line)
            if (ld != []):
                ret.append(ld)
            line = file_object.readline()
    return ret


def showLoss(inputs):
    for infile in inputs:
        filepth = infile.name
        print(filepth)
        data = parseLogEpochs(filepth)
        x = [float(elem[0]) for elem in data]
        y = [float(elem[1]) for elem in data]
        maxy = max(y)
        af = [float(elem[2]) * maxy for elem in data]
        cl = [random.random() for x in range(3)]
        plt.plot(x, y, color=cl)
        plt.plot(x, af, linestyle='dashed', color=cl)
    lgnd_lst = list()
    for e in inputs:
        lgnd_lst += [e.name.split("/")[0]]
        lgnd_lst += ["annealing factor"]

    plt.legend(lgnd_lst)
    plt.ylabel = "Loss"
    plt.xlabel = "Epoch"
    plt.show()


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', action='store_true', default=False, help='enables CUDA training [default: False]')
    parser.add_argument('--inputs', type=argparse.FileType('r'), nargs='+')
    args = parser.parse_args()
    showLoss(args.inputs)