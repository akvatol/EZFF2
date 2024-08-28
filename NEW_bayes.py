import numpy as np
import pandas as pd
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)

from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df
from ax import OptimizationConfig
from botorch.acquisition import (
    qExpectedImprovement,
    qUpperConfidenceBound,
    qKnowledgeGradient,
    qNoisyExpectedImprovement
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement as qNEI
from botorch.models import SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
import random

class CustomMetric(Metric):
    def fetch_trial_data(self, trial):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            val = custom_loss(params)
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "mean": val,
                "sem": 0.0,
                "trial_index": trial.index,
            })
        return Data(df=pd.DataFrame.from_records(records))

def optimize(search_space_dict, num_sobol_trials, num_botorch_trials, custom_runner):
    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=param_name, parameter_type=ParameterType.FLOAT,
                lower=param_range[0], upper=param_range[1]
            )
            for param_name, param_range in search_space_dict.items()
        ]
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=CustomMetric(name="custom_loss"),
            minimize=True,
        ),
    )

    exp = Experiment(
        name="multi_optimizer_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=custom_runner,
    )

    sobol_step = GenerationStep(
        model=Models.SOBOL,
        num_trials=num_sobol_trials,
    )

    surrogate_models = [SingleTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP]
    acquisition_functions = [qExpectedImprovement, qUpperConfidenceBound, qKnowledgeGradient, qNEI]

    botorch_step = GenerationStep(
        model=Models.BOTORCH_MODULAR,
        num_trials=-1,
        model_kwargs={
            "surrogate": Surrogate(random.choice(surrogate_models)),
            "botorch_acqf_class": random.choice(acquisition_functions),
        },
    )

    generation_strategy = GenerationStrategy(steps=[sobol_step, botorch_step])

    for i in range(num_sobol_trials + num_botorch_trials):
        if i == num_sobol_trials:
            generation_strategy.current_step.model_kwargs = {
                "surrogate": Surrogate(random.choice(surrogate_models)),
                "botorch_acqf_class": random.choice(acquisition_functions),
            }

        generator_run = generation_strategy.gen(experiment=exp)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

        if i >= num_sobol_trials and (i - num_sobol_trials + 1) % 3 == 0:
            generation_strategy.current_step.model_kwargs = {
                "surrogate": Surrogate(random.choice(surrogate_models)),
                "botorch_acqf_class": random.choice(acquisition_functions),
            }

    df = exp.fetch_data().df
    best_arm_name = df.loc[df['mean'].idxmin()]['arm_name']
    best_parameters = exp.arms_by_name[best_arm_name].parameters
    best_parameters_dict = {p_name: best_parameters[p_name] for p_name in search_space_dict.keys()}

    return best_parameters_dict

def parametrize(
    *,
    error_function,
    template_file,
    variable_bounds_file,
    num_sobol_trials,
    num_botorch_trials,
):
    ff_template = ffio.read_forcefield_template(template_file)
    variable_bounds, num_variables, variable_names = read_variable(
        filename=variable_bounds_file
    )

    search_space_dict = {name: bounds for name, bounds in zip(variable_names, variable_bounds)}

    def custom_loss(parameters):
        param_dict = dict(zip(variable_names, parameters.values()))
        return error_function(param_dict, ff_template)

    class CustomRunner(SyntheticRunner):
        def run(self, trial):
            trial_metadata = {"name": str(trial.index)}
            arm_names = []
            mean_values = []

            for arm_name, arm in trial.arms_by_name.items():
                params = arm.parameters
                val = custom_loss(params)
                arm_names.append(arm_name)
                mean_values.append(val)

            df = pd.DataFrame(
                {
                    "arm_name": arm_names,
                    "metric_name": "custom_loss",
                    "mean": mean_values,
                    "sem": 0.0,
                    "trial_index": trial.index,
                }
            )

            data = Data(df=df)
            trial.experiment.attach_data(data)
            return trial_metadata

    best_parameters_dict = optimize(
        search_space_dict,
        num_sobol_trials=num_sobol_trials,
        num_botorch_trials=num_botorch_trials,
        custom_runner=CustomRunner(),
    )

    return best_parameters_dict
