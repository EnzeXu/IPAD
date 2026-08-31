from ._utils import load_argparse, print_argparse

from ._lotka_volterra import LotkaVolterra
from ._sir import SIR

ODE_DICT = {
    "Lotka_Volterra": LotkaVolterra,
    "SIR": SIR,
}


def get_dataset(timestring=None):
    args, parser = load_argparse(timestring)
    print_argparse(args, parser)
    ode = ODE_DICT.get(args.task)(args)
    return ode


if __name__ == "__main__":
    ode = get_dataset()
    ode.build()
