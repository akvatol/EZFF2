# General import
import pandas as pd
import numpy as np

# EZFF utils import
from . import ffio
from ezff.ffio import generate_forcefield as gen_ff

# PyMOO
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.termination import NoTermination
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem
from joblib import Parallel, delayed


__version__ = "1.0.3"  # Update setup.py if version changes


def _get_mins_maxs(variable_bounds: dict):
    var_min = [min(i) for i in variable_bounds.values()]
    var_max = [max(i) for i in variable_bounds.values()]
    return var_min, var_max


def read_variable(
    filename: str,
) -> tuple[dict[str, tuple[float, float]], int, list[str]]:
    """Read permissible lower and upper bounds for decision variables used in force fields optimization.

    Args:
        filename (str): Name of text file listing bounds for each decision variable that must be optimized.

    Returns:
        tuple[dict[str, tuple[float, float]], int, list[str]]: A tuple containing the variable bounds,
            the number of variables, and the list of variable names.
    """
    variable_bounds = ffio.read_variable_bounds(filename)
    num_variables = len(variable_bounds)
    variable_names = list(variable_bounds.keys())

    return (variable_bounds, num_variables, variable_names)


def define_problem(
    *,
    num_errors: int,
    num_variables: int,
    variable_bounds: str,
) -> Problem:
    """
    Define a PyMOO problem instance.

    Returns:
    Problem: An instance of the MyProblem class.
    """

    class MyProblem(Problem):
        def __init__(selfself, **kwargs):
            xl, xu = _get_mins_maxs(variable_bounds)
            super().__init__(
                n_var=num_variables,
                n_obj=num_errors,
                n_ieq_constr=0,
                xl=xl,
                xu=xu,
                **kwargs,
            )

    return MyProblem()


class MyCallback(Callback):
    """
    A custom callback class for PyMOO.
    """

    def __init__(self, folder_path: str = None) -> None:
        """
        Initialize the callback instance.

        Parameters:
        folder_path (str, optional): The folder path to save the data. Defaults to None.
        """
        super().__init__()
        self.counter = 0
        self.folder_path = folder_path

    def notify(self, X, F) -> None:
        """
        Notify the callback of a new iteration.

        Parameters:
        algorithm: The PyMOO algorithm instance.
        """
        self.counter += 1
        path = (
            f"{self.folder_path}/{self.counter}"
            if self.folder_path
            else f"{self.counter}"
        )

        pd.DataFrame(F).to_csv(f"{path}_errors.csv", index=False)
        pd.DataFrame(X).to_csv(f"{path}_values.csv", index=False)


def parametrize(
    callback=None,
    *,
    algorithm,
    n_gen,
    num_errors,
    error_function,
    template_file,
    variable_bounds_file,
    n_jobs,
    seed=1,
):
    ff_template = ffio.read_forcefield_template(template_file)
    variable_bounds, num_variables, variable_names = read_variable(
        filename=variable_bounds_file
    )

    problem = define_problem(
        num_errors=num_errors,
        variable_bounds=variable_bounds,
        num_variables=num_variables,
    )
    algorithm.setup(problem, termination=NoTermination())
    np.random.seed(seed)
    if not callback:
        callback = MyCallback()

    for i in range(n_gen):
        pop = algorithm.ask()
        X = pop.get("X")
        data = Parallel(n_jobs=n_jobs)(
            delayed(error_function)(dict(zip(variable_names, x)), ff_template)
            for x in X
        )
        F = np.array(data)

        callback.notify(X=X, F=F)
        static = StaticProblem(problem, F=F)
        Evaluator().eval(static, pop)

        # returned the evaluated individuals which have been evaluated or even modified
        algorithm.tell(infills=pop)

        # do same more things, printing, logging, storing or even modifying the algorithm object
        print(algorithm.n_gen)
    res = algorithm.result()
    return res


def generate_forcefield(template_string, parameters, FFtype=None, outfile=None):
    forcefield = gen_ff(
        template_string, parameters, FFtype=FFtype, outfile=outfile, MD="GULP"
    )
    return forcefield
