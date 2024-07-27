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
        name="custom_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=custom_runner,
    )

    print(f"Running Sobol initialization trials...")
    sobol = Models.SOBOL(search_space=exp.search_space)

    for i in range(num_sobol_trials):
        generator_run = sobol.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

    for i in range(num_botorch_trials):
        print(f"Running BO trial {i + num_sobol_trials + 1}/{num_sobol_trials + num_botorch_trials}...")
        gpei = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())
        generator_run = gpei.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

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
