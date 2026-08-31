from ._utils import load_argparse, print_argparse

from ._lotka_volterra import LotkaVolterra
from ._lorenz import Lorenz
from ._sir import SIR
from ._damped_pendulum import DampedPendulum

ODE_DICT = {
    "Lotka_Volterra": LotkaVolterra,
    "Lorenz": Lorenz,
    "SIR": SIR,
    "Damped_Pendulum": DampedPendulum,
}


def get_dataset(timestring=None):
    args, parser = load_argparse(timestring)
    print_argparse(args, parser)
    ode = ODE_DICT.get(args.task)(args)
    return ode


if __name__ == "__main__":
    ode = get_dataset()
    ode.build()
