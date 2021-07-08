import datetime
import timeit


def get_argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-latents', type=int, default=64,
                        help='size of the latent embedding [default: 64]')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--annealing-epochs', type=int, default=200, metavar='N',
                        help='number of epochs to anneal KL for [default: 200]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default: 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--lambda-image', type=float, default=1.,
                        help='multipler for image reconstruction [default: 1]')
    parser.add_argument('--lambda-text', type=float, default=10.,
                        help='multipler for text reconstruction [default: 10]')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training [default: False]')
    return parser


def get_executionName():
    from time import localtime, strftime
    st = strftime("%d%b%Y%H%M%S", localtime())
    return st


def saveargs(id, args):
    st = id + "\n"
    for key in vars(args):
        st += key + " : " + str(args.__dict__.get(key)) + "\n"
    text_file = open("./" + id + "/params.txt", "wt")
    n = text_file.write(st)
    text_file.close()


def logDetailedEpch(id, st):
    import os
    if not os.path.exists("./" + id + "/log_epochs_detailed.txt"):
        mode = "w"
    else:
        mode = "a"
    text_file = open("./" + id + "/log_epochs_detailed.txt", mode)
    text_file.write(st + "\n")
    text_file.close()


def logEpoch(id, st):
    import os
    if not os.path.exists("./" + id + "/log_epochs.txt"):
        mode = "w"
    else:
        mode = "a"
    text_file = open("./" + id + "/log_epochs.txt", mode)
    text_file.write(st + "\n")
    text_file.close()


def append_to_params_log(id, st):
    text_file = open("./" + id + "/params.txt", "a")
    n = text_file.write(st)
    text_file.close()


def logNetStruct(id, model):
    st = "model structure (vars dump):\n"
    for key in vars(model):
        st += key + " : " + str(model.__dict__.get(key)) + "\n"
    text_file = open("./" + id + "/structure.txt", "wt")
    n = text_file.write(st)
    text_file.close()


def logCode(id, model):
    import inspect
    st = ""
    for elem in model._modules.values():
        st += type(elem).__name__ + "\n"
        tmp = inspect.getsource(getattr(elem, "forward"))
        st += tmp.replace("\n", "\n    ")
        st += "\r\n"
    text_file = open("./" + id + "/forwards code dump.txt", "wt")
    n = text_file.write(st)
    text_file.close()


class Timer:

    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self._start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))