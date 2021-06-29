import re
import matplotlib.pyplot as plt

def parseLogEpochs(filepth):
    ret=list()
    with open(filepth, 'r') as file_object:
        line=file_object.readline()
        while(line):
            ld = re.findall('\d+\.\d+|\d+', line)
            ret.append(ld)
            line=file_object.readline()
    return ret


def showLoss(inputs):
    for infile in inputs:
        filepth=infile.name
        print(filepth)
        data=parseLogEpochs(filepth)
        print(data)
        x=[elem[0] for elem in data ]
        y=[float(elem[1]) for elem in data ]
        plt.plot(x,y)
    plt.legend([e.name.split("/")[0] for e in inputs])
    plt.ylabel = "Loss"
    plt.xlabel = "Epoch"
    plt.show()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', action='store_true', default=False, help='enables CUDA training [default: False]')
    parser.add_argument('--inputs', type=argparse.FileType('r'), nargs='+')
    args=parser.parse_args()
    showLoss(args.inputs)


