from pyswmm import Simulation, Nodes, Links, RainGages, Subcatchments
import os.path
import csv
import re
import strategoutil as sutil
import datetime
import sys
import yaml
import weather_forecast_generation as weather
import subprocess # from stompc
import shutil # from stompc
import os # from stompc

def swmm_control(swmm_inputfile, orifice_id, basin_id, junction_id, time_step, csv_file_basename, controllers,
                 period, horizon, rain_data_file, weather_forecast_path, uncertainty):
    """
    Implement the control strategy from uppaal stratego to swmm
    requires:
        swmm_inputfile: path
        orifice_id: string
        open_settings: percentage of orifice opening, float, from 0 to 1
        basin_id: string, upstream basin ID
        time_step: float
        csv_file_basename: path

    returns:
        one csv files with the results
    """
    time_series = []
    water_depth_basins = []
    orifice_settings = []
    rain = []
    weather_forecast_low = []
    weather_forecast_high = []
    weather_forecast_int = []
    water_depth_junctions = []  # add junctions

    with Simulation(swmm_inputfile) as sim:
        sys.stdout.write('\n')
        i = 0
        interval = sim.end_time - sim.start_time
        duration = interval.total_seconds() / time_step
        print_progress_bar(i, duration, "progress")

        sus = []
        orifices = []
        #junctions = []  #add junctions

        for j in range(len(controllers)):
            sus.append(Nodes(sim)[f"{basin_id}{j+1}"])
            orifices.append(Links(sim)[f"{orifice_id}{j+1}"])
            orifice_settings.append([])
            water_depth_basins.append([])
            #junctions.append(Nodes(sim)[f"{junction_id}{j+2}"]) #junction4 which reflects pond1,2,3
            #water_depth_junctions.append([])
        #j2 = Nodes(sim)["J2"]

        ca = Subcatchments(sim)["S1"]
        sim.step_advance(time_step)
        current_time = sim.start_time

        for controller, orifice, orifice_setting, su, water_depth_basin in \
                zip(controllers, orifices, orifice_settings, sus, water_depth_basins):
            orifice.target_setting = get_control_strategy(su.depth, current_time, controller,
                                                          period, horizon, rain_data_file,
                                                          weather_forecast_path, uncertainty)
            #orifice_setting.append(1.75*orifice.target_setting + 2)  # Offset for printing purposes
            orifice_setting.append(orifice.target_setting)
            water_depth_basin.append(su.depth)
            #water_depth_junction.append(junction.depth)

        rain_low, rain_high, rain_int = get_weather_forecast_result(weather_forecast_path)
        weather_forecast_low.append(rain_low)
        weather_forecast_high.append(rain_high)
        weather_forecast_int.append(rain_int)
        time_series.append(sim.start_time)
        rain.append(0)
        total_precipitation = 0

        for step in sim:
            current_time = sim.current_time
            time_series.append(current_time)

            i = i + 1
            print_progress_bar(i, duration, "progress")

            rain.append(ca.statistics.get('precipitation') - total_precipitation)
            total_precipitation = ca.statistics.get('precipitation')

            for controller, orifice, orifice_setting, su, water_depth_basin in \
                    zip(controllers, orifices, orifice_settings, sus, water_depth_basins):
                water_depth_basin.append(su.depth)
                #water_depth_junction.append(junction.depth)

                # Set the control parameter
                orifice.target_setting = get_control_strategy(su.depth, current_time, controller,
                                                              period, horizon, rain_data_file,
                                                              weather_forecast_path, uncertainty)
                #orifice_setting.append(1.75*orifice.target_setting + 2)  # Offset for printing purposes
                orifice_setting.append(orifice.target_setting)

            rain_low, rain_high, rain_int = get_weather_forecast_result(weather_forecast_path)
            weather_forecast_low.append(rain_low)
            weather_forecast_high.append(rain_high)
            weather_forecast_int.append(rain_int)

    i = i + 1
    print_progress_bar(i, duration, "progress")
    dirname = os.path.dirname(swmm_inputfile)
    output_csv_file = os.path.join(dirname, csv_file_basename + "." + "csv")
    with open(output_csv_file, "w") as f:
        writer = csv.writer(f)
        top_row = ["time", "rain", "forecast_low", "forecast_high", "forecast_int"]
        for j in range(len(controllers)):
            top_row.extend([f"water_depth_basin{j+1}"])
        for j in range(len(controllers)):
            top_row.extend([f"orifice_setting{j+1}"])
        #top_row.extend([f"water_depth_junction{j+2}"])
        writer.writerow(top_row)

        for line_index in range(len(time_series)):
            line = [time_series[line_index].strftime('%Y-%m-%d %H:%M')]
            line.append(rain[line_index])
            line.append(weather_forecast_low[line_index])
            line.append(weather_forecast_high[line_index])
            line.append(weather_forecast_int[line_index])
            for water_depth_basin in water_depth_basins:
                line.append(water_depth_basin[line_index])
            for orifice_setting in orifice_settings:
                line.append(orifice_setting[line_index])
            #line.append(water_depth_junction[line_index])
            writer.writerow(line)

def get_control_strategy(current_water_level, current_time, controller, period, horizon,
                         rain_data_file, weather_forecast_path, uncertainty):
    controller.controller.update_state({'w': current_water_level * 100}) #  Conversion from m to cm.
    print(current_water_level)
    control_setting = controller.run_single(period, horizon, start_date=current_time,
                                            historical_rain_data_path=rain_data_file,
                                            weather_forecast_path=weather_forecast_path,
                                            uncertainty=uncertainty)

    return control_setting


def get_weather_forecast_result(weather_forecast_path):
    with open(weather_forecast_path, "r") as f:
        weather_forecast = csv.reader(f, delimiter=',', quotechar='"')
        headers = next(weather_forecast)
        first_data = next(weather_forecast)

        return int(first_data[0]), int(first_data[1]), float(first_data[4])


def print_progress_bar(i, max, post_text):
    """
    Print a progress bar to sys.stdout.

    Subsequent calls will override the previous progress bar (given that nothing else has been
    written to sys.stdout).

    From `<https://stackoverflow.com/a/58602365>`_.

    :param i: The number of steps already completed.
    :type i: int
    :param max: The maximum number of steps for process to be completed.
    :type max: int
    :param post_text: The text to display after the progress bar.
    :type post_text: str
    """
    n_bar = 20  # Size of progress bar.
    j = i / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}")
    sys.stdout.flush()

def insert_rain_data_file_path(swmm_inputfile, rain_data_file):
    """
    Insert the provided rain data file path into the swmm model.

    :param str swmm_inputfile: swmm model path
    :param str rain_data_file: rain data file path
    """
    with open(swmm_inputfile, "r+") as f:
        file_content = f.read()
        new_line = "long_term_rainfallgauge5061 FILE \"" + rain_data_file + "\""
        file_content = re.sub(r"long_term_rainfallgauge5061 FILE \"[^\"]*\"", new_line,
                              file_content, count=1)
        f.seek(0)
        f.write(file_content)
        f.truncate()


def insert_paths_in_uppaal_model(uppaal_model, weather_forecast_path, libtable_path):
    """
    Insert the provided weather forecast path into the uppaal model.

    :param str uppaal_model: uppaal model path
    :param str weather_forecast_path: weather forecast path
    :param str libtable_path: libtable.so path
    """
    with open(uppaal_model, "r+") as f:
        file_content = f.read()
        new_line = "const int file_id = table_read_csv(\"" + weather_forecast_path + "\""
        file_content = re.sub(r"const int file_id = table_read_csv\(\"[^\"]*\"", new_line,
                              file_content, count=1)
        new_line = "import \"" + libtable_path + "\""
        file_content = re.sub(r"import \"[^\"]*\"", new_line, file_content, count=1)
        f.seek(0)
        f.write(file_content)
        f.truncate()


def get_int_tuples(text):
    """
    Convert Stratego simulation output to list of tuples (int, int).

    :param text: The input string containing the Uppaal Stratego output.
    :type text: str
    :return: A list of tuples (int, int).
    :rtype: list
    """
    string_tuples = re.findall(r"\((\d+),(\d+)\)", text)
    if string_tuples is None:
        raise RuntimeError(
            "Output of Stratego has not the expected format of a list of tuples. Please check the "
            "output manually for error messages: \n" + text)
    int_tuples = [(int(t[0]), int(t[1])) for t in string_tuples]
    return int_tuples


def get_float_tuples(text):
    """
    Convert Stratego simulation output to list of tuples (float, float).

    :param text: The input string containing the Uppaal Stratego output.
    :type text: str
    :return: A list of tuples (float, float).
    :rtype: list
    """
    float_re = r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
    pattern = r"\(" + float_re + "," + float_re + r"\)"
    string_tuples = re.findall(pattern, text)
    if string_tuples is None:
        raise RuntimeError(
            "Output of Stratego has not the expected format of a list of tuples. Please check the "
            "output manually for error messages: \n" + text)
    float_tuples = [(float(t[0]), float(t[4])) for t in string_tuples]
    return float_tuples


def extract_state(text, var, control_period):
    """
    Extract the state from the Uppaal Stratego output at the end of the simulated control period.

    :param text: The input string containing the Uppaal Stratego output.
    :type text: str
    :param var: The variable name.
    :type var: str
    :param control_period: The interval duration after which the controller can change the control
        setting, given in Uppaal Stratego time units.
    :type control_period: int
    :return: The value of the variable at the end of *control_period*.
    :rtype: float
    """
    float_re = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
    pattern = var + r":\n\[0\]:( \(" + float_re + "," + float_re + r"\))*"
    result = re.search(pattern, text)
    if result is None:
        raise RuntimeError(
            "Output of Stratego has not the expected format. Please check the output manually for "
            "error messages: \n" + text)
    float_tuples = get_float_tuples(result.group())
    x, y = 0.0, 0.0
    last_value = 0.0
    p = 1
    for t in float_tuples:
        while p * control_period < t[0]:
            last_value = y + (p * control_period - x) * (t[1] - y) / (t[0] - x)
            p += 1
        x = t[0]
        y = t[1]
    return last_value


def get_duration_action(tuples, max_time=None):
    """
    Get tuples (duration, action) from tuples (time, variable) resulted from simulate query.
    """
    # TODO: This method is currently not used in any of the classes. Can we safely remove this?
    result = []
    if len(tuples) == 1:  # Can only happen if always variable == 0.
        result.append((max_time, 0))
    elif len(tuples) == 2:  # Can only happen of always variable != 0.
        action = tuples[1][1]
        result.append((max_time, action))
    else:
        for i in range(1, len(tuples)):
            duration = tuples[i][0] - tuples[i - 1][0]
            action = tuples[i][1]
            if duration > 0:
                result.append((duration, action))

    return result


def insert_to_modelfile(model_file, tag, inserted):
    """
    Replace tag in model file by the desired text.

    :param model_file: The file name of the model.
    :type model_file: str
    :param tag: The tag to replace.
    :type tag: str
    :param inserted: The value to replace the tag with.
    :type inserted: str
    """
    with open(model_file, "r+") as f:
        model_text = f.read()
        text = model_text.replace(tag, inserted, 1)
        f.seek(0)
        f.write(text)
        f.truncate()


def array_to_stratego(arr):
    """
    Convert python array string to C style array used in UPPAAL Stratego.
    NB, does not include ';' in the end.

    :param arr: The array string to convert.
    :type arr: str
    :return: An array string where ``"["`` and ``"]"`` are replaced by ``"{"`` and ``"}"``,
        respectively.
    :rtype: str
    """
    arrstr = str(arr)
    arrstr = str.replace(arrstr, "[", "{", 1)
    arrstr = str.replace(arrstr, "]", "}", 1)
    return arrstr


def merge_verifyta_args(cfg_dict):
    """
    Concatenate and format a string of verifyta arguments given by the configuration dictionary.

    :param cfg_dict: The configuration dictionary.
    :type cfg_dict: dict
    :return: String containing all arguments from the configuration dictionary.
    :rtype: str
    """
    args = ""
    for k, v in cfg_dict.items():
        if v is not None:
            args += " --" + k + " " + str(v)
        else:
            args += " --" + k
    return args[1:]


def check_tool_existence(name):
    """
    Check whether 'name' is on PATH and marked executable.

    From `<https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script>`_.

    :param name: the name of the tool.
    :type name: str
    :return: True when the tool is found and executable, false otherwise.
    :rtype: bool
    """
    return shutil.which(name) is not None


def run_stratego(model_file, query_file="", learning_args=None, verifyta_command="verifyta"):
    """
    Run command line version of Uppaal Stratego.

    :param model_file: The file name of the model.
    :type model_file: str
    :param query_file: The file name of the query.
    :type query_file: str
    :param learning_args: Dictionary containing the learning parameters and their values. The
        learning parameter names should be those used in the command line interface of Uppaal
        Stratego. You can also include non-learning command line parameters in this dictionary.
        If a non-learning command line parameter does not take any value, include the empty
        string ``""`` as value.
    :type learning_args: dict
    :param verifyta_command: The command name for running Uppaal Stratego at the user's machine.
    :type verifyta_command: str
    :return: The output as produced by Uppaal Stratego.
    :rtype: str
    """
    learning_args = {} if learning_args is None else learning_args
    args = {
        "verifyta": verifyta_command,
        "model": model_file,
        "query": query_file,
        "config": merge_verifyta_args(learning_args)
    }
    args_list = [v for v in args.values() if v != ""]
    task = " ".join(args_list)

    process = subprocess.Popen(task, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = process.communicate()
    result = [r.decode("utf-8") for r in result]
    return result


def successful_result(text):
    """
    Verify whether the stratego output is based on the successful synthesis of a strategy.

    :param text: The output generated by Uppaal Stratego.
    :type text: str
    :return: Whether Uppaal Stratego has successfuly ran all queries.
    :rtype: bool
    """
    result = re.search("Formula is satisfied", text)
    return result is not None


class StrategoController:
    """
    Controller class to interface with UPPAAL Stratego through python.

    :param model_template_file: The file name of the template model.
    :type model_template_file: str
    :param model_cfg_dict: Dictionary containing pairs of state variable name and its initial
        value. The state variable name should match the tag name in the template model.
    :type model_cfg_dict: dict
    :param cleanup: Whether or not to clean up the temporarily simulation file after being used.
    :type cleanup: bool
    :ivar states: Dictionary containing the current state of the system, where a state is a pair of
        variable name and value. It is initialized with the values from *model_cfg_dict*.
    :vartype states: bool
    :ivar tagRule: The rule for each tag in the template model. Currently, the rul is set to be
        ``"//TAG_{}"``. Therefore, tags in the template model should be ``"//TAG_<variable name>"``,
        where ``<variable name>`` is the global name of the variable.
    :vartype tagRule: str
    """

    def __init__(self, model_template_file, model_cfg_dict, cleanup=True):
        self.template_file = model_template_file
        self.simulation_file = model_template_file.replace(".xml", "_sim.xml")
        self.cleanup = cleanup  # TODO: this variable seems to be not used. Can it be safely removed?
        self.states = model_cfg_dict.copy()
        self.tagRule = "//TAG_{}"

    def init_simfile(self):
        """
        Make a copy of a template file where data of specific variables is inserted.
        """
        shutil.copyfile(self.template_file, self.simulation_file)

    def remove_simfile(self):
        """
        Clean created temporary files after the simulation is finished.
        """
        os.remove(self.simulation_file)

    def debug_copy(self, debug_file):
        """
        Copy UPPAAL simulationfile.xml file for manual debug in Stratego.

        :param debug_file: The file name of the debug file.
        :type debug_file: str
        """
        shutil.copyfile(self.simulation_file, debug_file)

    def update_state(self, new_values):
        """
        Update the state of the MPC controller.

        :param new_values: Dictionary containing new values for the state variables.
        :type new_values: dict
        """
        for name, value in new_values.items():
            self.states.update({name: value})

    def insert_state(self):
        """
        Insert the current state values of the variables at the appropriate position in the
        simulation \*.xml file indicated by the :py:attr:`tagRule`.
        """
        for name, value in self.states.items():
            tag = self.tagRule.format(name)
            insert_to_modelfile(self.simulation_file, tag, str(value))

    def get_var_names_as_string(self):
        """
        Print the names of the state variables separated by a ','.

        :return: All the variable names joined together with a ','.
        :rtype: str
        """
        separator = ","
        return separator.join(self.states.keys())

    def get_state_as_string(self):
        """
        Print the values of the state variables separated by a ','.

        :return: All the variable values joined together with a ','.
        :rtype: str
        """
        separator = ","
        values_as_string = [str(val) for val in self.states.values()]
        return separator.join(values_as_string)

    def get_state(self, key):
        """
        Get the current value of the provided state variable.

        :param key: The state variable name.
        :type key: str
        :return: The currently stored value of the state variable.
        :rtype: int or float
        """
        return self.states.get(key)

    def get_states(self):
        """
        Get the current states.

        :return: The current state dictionary.
        :rtype: dict
        """
        return self.states

    def run(self, query_file="", learning_args=None, verifyta_command="verifyta"):
        """
        Runs verifyta with requested queries and parameters that are either part of the \*.xml model
        file or explicitly specified.

        :param query_file: The file name of the query file where the queries are written to.
        :type query_file: str
        :param learning_args: Dictionary containing the learning parameters and their values. The
            learning parameter names should be those used in the command line interface of Uppaal
            Stratego. You can also include non-learning command line parameters in this dictionary.
            If a non-learning command line parameter does not take any value, include the empty
            string ``""`` as value.
        :type learning_args: dict
        :param verifyta_command: The command name for running Uppaal Stratego at the user's machine.
        :type verifyta_command: str
        :return: The output generated by Uppaal Stratego.
        :rtype: str
        """
        learning_args = {} if learning_args is None else learning_args
        output = run_stratego(self.simulation_file, query_file, learning_args, verifyta_command)
        return output[0]






class MPCSetupPond(sutil.SafeMPCSetup):

    def __init__(self, model_template_file, output_file_path=None, query_file="",
                 model_cfg_dict=None, learning_args=None, verifyta_command="verifyta",
                 external_simulator=False, action_variable=None, debug=False):
        self.model_template_file = model_template_file
        self.output_file_path = output_file_path
        self.query_file = query_file
        self.model_cfg_dict = {} if model_cfg_dict is None else model_cfg_dict
        self.learning_args = {} if learning_args is None else learning_args
        self.verifyta_command = verifyta_command
        self.external_simulator = external_simulator
        if external_simulator and action_variable not in model_cfg_dict.keys():
            raise RuntimeError(
                f"The provided action variable {action_variable} is not defined as a model variable"
                f"in the model configuration.")
        self.action_variable = action_variable
        self.debug = debug
        self.controller = StrategoController(self.model_template_file, self.model_cfg_dict)

    def step_without_sim(self, control_period, horizon, duration, step, **kwargs):
        """
        Perform a step in the basic MPC scheme without the simulation of the synthesized strategy.

        :param control_period: The interval duration after which the controller can change the
            control setting, given in Uppaal Stratego time units.
        :type control_period: int
        :param horizon: The interval duration for which Uppaal stratego synthesizes a control strategy
            each MPC step. Is given in the number of control periods.
        :type horizon: int
        :param duration: The number of times (steps) the MPC scheme should be performed, given as
            the number of control periods. Is only forwarded to
            :meth:`~MPCsetup.perform_at_start_iteration`.
        :type duration: int
        :param step: The current iteration step in the basic MPC loop.
        :type step: int
        :param kwargs: Any additional parameters are forwarded to
            :meth:`~MPCsetup.perform_at_start_iteration`.
        :return: The output generated by Uppaal Stratego.
        :rtype: str
        """
        # Perform some customizable preprocessing at each step.
        self.perform_at_start_iteration(control_period, horizon, duration, step, **kwargs)

        # At each MPC step we want a clean template copy to insert variables.
        self.controller.init_simfile()

        # Insert current state into simulation template.
        self.controller.insert_state()

        # To debug errors from verifyta one can save intermediate simulation file.
        if self.debug:
            self.controller.debug_copy(self.model_template_file.replace(".xml", "_debug.xml"))

        # Create the new query file for the next step.
        final = horizon * control_period + self.controller.get_state("t")
        self.create_query_file(horizon, control_period, final)

        # Run a verifyta query to simulate optimal strategy.
        result = self.run_verifyta(horizon, control_period, final)

        return result

    def run_single(self, control_period, horizon, **kwargs):
        """
        Run the basic MPC scheme a single step where a single controller strategy is calculated,
        where the strategy synthesis looks the horizon ahead, and continues for the duration of the
        experiment.

        The control period is in Uppaal Stratego time units. Horizon have control period as time
        unit.

        :param control_period: The interval duration after which the controller can change the
            control setting, given in Uppaal Stratego time units.
        :type control_period: int
        :param horizon: The interval duration for which Uppaal stratego synthesizes a control strategy
            each MPC step. Is given in the number of control periods.
        :type horizon: int
        :param `**kwargs`: Any additional parameters are forwarded to
            :meth:`~MPCsetup.perform_at_start_iteration`.
        :return: The control action chosen for the first control period.
        """
        if not check_tool_existence(self.verifyta_command):
            raise RuntimeError(
                f"Cannot find the supplied verifyta command: {self.verifyta_command}")

        result = self.step_without_sim(control_period, horizon, 1, 0, **kwargs)
        chosen_action = self.extract_control_action_from_stratego(result)

        return chosen_action

    def run(self, control_period, horizon, duration, **kwargs):
        """
        Run the basic MPC scheme where the controller can changes its strategy once every period,
        where the strategy synthesis looks the horizon ahead, and continues for the duration of the
        experiment.

        The control period is in Uppaal Stratego time units. Both horizon and duration have control
        period as time unit.

        :param control_period: The interval duration after which the controller can change the
            control setting, given in Uppaal Stratego time units.
        :type control_period: int
        :param horizon: The interval duration for which Uppaal stratego synthesizes a control strategy
            each MPC step. Is given in the number of control periods.
        :type horizon: int
        :param duration: The number of times (steps) the MPC scheme should be performed, given as
            the number of control periods.
        :type duration: int
        :param `**kwargs`: Any additional parameters are forwarded to
            :meth:`~MPCsetup.perform_at_start_iteration`.
        """
        # Print the variable names and their initial values.
        self.print_state_vars()
        self.print_state()

        if not check_tool_existence(self.verifyta_command):
            raise RuntimeError(
                f"Cannot find the supplied verifyta command: {self.verifyta_command}")

        for step in range(duration):
            # Only print progress to stdout if results are printed to a file.
            if self.output_file_path:
                print_progress_bar(step, duration, "progress")

            result = self.step_without_sim(control_period, horizon, duration, step, **kwargs)

            if self.external_simulator:
                # An external simulator is used to generate the new 'true' state.
                chosen_action = self.extract_control_action_from_stratego(result)
                new_state = self.run_external_simulator(chosen_action, control_period, step,
                                                        **kwargs)
                self.controller.update_state(new_state)

            else:
                # Extract the state from Uppaal results. This requires that the query file also
                # includes a simulate query (see default query generator).
                self.extract_states_from_stratego(result, control_period)

            # Print output.
            self.print_state()
        if self.output_file_path:
            print_progress_bar(duration, duration, "finished")

    def run_verifyta(self, horizon, control_period, final, *args, **kwargs):
        """
        Run verifyta with the current data stored in this class.

        It verifies whether Stratego has successfully synthesized a strategy. If not, it will create
        an alternative query file and run Stratego again.

        Overrides :meth:`~MPCsetup.run_verifyta()` in :class:`~MPCsetup`.

        :param horizon: The interval duration for which Uppaal stratego synthesizes a control strategy
            each MPC step. Is given in the number of periods.
        :type horizon: int
        :param control_period: The interval duration after which the controller can change the
            control setting, given in Uppaal Stratego time units.
        :type control_period: int
        :param final: The time that should be reached by the synthesized strategy, given in Uppaal
            Stratego time units. Most likely this will be current time + *horizon* x *period*.
        :type final: int
        :param `*args`: Is not used in this method; it is included here to safely override the
            original method.
        :param `**kwargs`: Is not used in this method; it is included here to safely override the
            original method.
        """
        result = self.controller.run(query_file=self.query_file, learning_args=self.learning_args,
                                     verifyta_command=self.verifyta_command)

        if not successful_result(result):
            self.create_alternative_query_file(horizon, control_period, final)
            result = self.controller.run(query_file=self.query_file, learning_args=self.learning_args,
                                         verifyta_command=self.verifyta_command)

        if self.controller.cleanup:
            self.controller.remove_simfile()
        return result

    def extract_states_from_stratego(self, result, control_period):
        """
        Extract the new state values from the simulation output of Stratego.

        The extracted values are directly saved in the :attr:`~MPCsetup.controller`.

        :param result: The output as generated by Uppaal Stratego.
        :type result: str
        :param control_period: The interval duration after which the controller can change the
            control setting, given in Uppaal Stratego time units.
        :type control_period: int
        """
        new_state = {}
        for var, value in self.controller.get_states().items():
            new_value = extract_state(result, var, control_period)
            if isinstance(value, int):
                new_value = int(new_value)
            new_state[var] = new_value
        self.controller.update_state(new_state)

    def extract_control_action_from_stratego(self, stratego_output):
        """
        Extract the chosen control action for the first control period from the simulation output
        of Stratego.

        :param stratego_output: The output as generated by Uppaal Stratego.
        :type stratego_output: str
        :return: The control action chosen for the first control period.
        :rtype: float
        """
        float_re = r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?"
        pattern = self.action_variable + r":\n\[0\]:( \(" + float_re + "," + float_re + r"\))*"
        result = re.search(pattern, stratego_output)
        if result is None:
            raise RuntimeError(
                "Output of Stratego has not the expected format. Please check the output manually "
                "for error messages: \n" + stratego_output)
        float_tuples = get_float_tuples(result.group())
        last_value = 0.0

        # The last tuple at time 0 represents the chosen control action.
        for t in float_tuples:
            if t[0] == 0:
                last_value = t[1]
            else:
                break
        return last_value

    def run_external_simulator(self, chosen_action, *args, **kwargs):
        """
        Run an external simulator to obtain the 'true' state after applying the synthesized control
        action for a single control period.

        This method should be overridden by the user. The method should return the new 'true' state
        as a dictionary containing pairs where the key is a variable name and the value is its new
        value.

        :param chosen_action: The synthesized control action for the first control period.
        :type chosen_action: int or float
        :return: The 'true' state of the system after simulation a single control period. The
            dictionary containings pairs of state variable name and their values. The state variable
            name should match the tag name in the template model.
        :rtype: dict
        """
        return {}

    def print_state_vars(self):
        """
        Print the names of the state variables to output file if provided. Otherwise, it will be
        printed to the standard output.
        """
        content = self.controller.get_var_names_as_string() + "\n"
        if self.output_file_path is None:
            sys.stdout.write(content)
        else:
            with open(self.output_file_path, "w") as f:
                f.write(content)

    def print_state(self):
        """
        Print the current state to output file if provided. Otherwise, it will be printed to the
        standard output.
        """
        content = self.controller.get_state_as_string() + "\n"
        if self.output_file_path is None:
            sys.stdout.write(content)
        else:
            with open(self.output_file_path, "a") as f:
                f.write(content)


    def create_query_file(self, horizon, period, final):
        """
        Create the query file for each step of the pond model. Current
        content will be overwritten.

        Overrides SafeMPCsetup.create_query_file().
        """
        with open(self.query_file, "w") as f:
            line1 = f"strategy opt = minE (2*st_c + c) [<={horizon}*{period}]: <> (t=={final})\n"
            f.write(line1)
            f.write("\n")
            line2 = f"simulate [<={period}+1; 1] {{ {self.controller.get_var_names_as_string()} " \
                    f"}} under opt\n"
            f.write(line2)

    def create_alternative_query_file(self, horizon, period, final):
        """
        Create an alternative query file in case the original query could not be satisfied by
        Stratego, i.e., it could not find a strategy. Current content will be overwritten.

        Overrides SafeMPCsetup.create_alternative_query_file().
        """
        with open(self.query_file, "w") as f:
            line1 = f"strategy opt = minE (2*st_c + c) [<={horizon}*{period}]: <> (t=={final})\n"
            f.write(line1)
            f.write("\n")
            line2 = f"simulate [<={period}+1; 1] {{ {self.controller.get_var_names_as_string()} " \
                    f"}} under opt\n"
            f.write(line2)


        #with open(self.query_file, "w") as f:
        #    line1 = f"strategy opt = minE (wmax) [<={horizon}*{period}]: <> (t=={final})\n"
        #    f.write(line1)
        #    f.write("\n")
        #    line2 = f"simulate 1 [<={period}+1] {{ {self.controller.get_var_names_as_string()} " \
        #            f"}} under opt\n"
        #    f.write(line2)

    def perform_at_start_iteration(self, controlperiod, horizon, duration, step, **kwargs):
        """
        Performs some customizable preprocessing steps at the start of each MPC iteration.

        Overrides SafeMPCsetup.perform_at_start_iteration().
        """
        current_date = kwargs["start_date"] + datetime.timedelta(hours=step)
        weather.create_weather_forecast(kwargs["historical_rain_data_path"],
                                        kwargs["weather_forecast_path"], current_date,
                                        horizon * controlperiod, kwargs["uncertainty"])


def main():
    # It is for path setting. This is OS dependent.
    this_file = os.path.realpath(__file__)
    base_folder = os.path.dirname(this_file)

    # Setting directory for swmm.
    swmm_folder = "swmm"  # swmm model located folder name
    swmm_inputfile = os.path.join(base_folder, swmm_folder, "swmm1.inp")
    assert (os.path.isfile(swmm_inputfile))  # if there is the file, go ahead!

    # We found the model. Now we have to include the correct path to the rain data into the model.
    rain_data_file = "swmm_5061.dat"  # Assumed to be in the same folder as the swmm model input file.
    rain_data_file = os.path.join(base_folder, swmm_folder, rain_data_file)
    insert_rain_data_file_path(swmm_inputfile, rain_data_file)

    # Finally we can specify other variables of swmm.
    orifice_id = "OR"
    basin_id = "SU"
    junction_id = "J"  # add junction
    num_basins = 3 # 3 ponds
    time_step = 60 * 60  # 60 seconds/min x 60 min/h -> 1 h
    swmm_results = "swmm_decentralized_results_test"

    # Now we locate the Uppaal folder and files.
    uppaal_folder_name = "uppaal_compo"
    uppaal_folder = os.path.join(base_folder, uppaal_folder_name)

    model_template_path_1 = os.path.join(uppaal_folder, "pond_compo_stream_1.xml")  # change
    model_template_path_2 = os.path.join(uppaal_folder, "pond_compo_stream_2.xml")  # change
    model_template_path_3 = os.path.join(uppaal_folder, "pond_compo_stream_3.xml")  # change

    query_file_path_1 = os.path.join(uppaal_folder, "pond_compo_stream_1_query.q")  # change
    query_file_path_2 = os.path.join(uppaal_folder, "pond_compo_stream_2_query.q") # change
    query_file_path_3 = os.path.join(uppaal_folder, "pond_compo_stream_3_query.q") # change

    model_config_path_1 = os.path.join(uppaal_folder, "pond_compo_stream_1_config.yaml") # change
    model_config_path_2 = os.path.join(uppaal_folder, "pond_compo_stream_2_config.yaml") # change
    model_config_path_3 = os.path.join(uppaal_folder, "pond_compo_stream_3_config.yaml") # change

    learning_config_path = os.path.join(uppaal_folder, "verifyta_decentralized_stream_config.yaml")
    weather_forecast_path = os.path.join(uppaal_folder, "decentralized_weather_forecast.csv")
    output_file_path_1 = os.path.join(uppaal_folder, "decentralized_stream_result_1.txt")
    output_file_path_2 = os.path.join(uppaal_folder, "decentralized_stream_result_2.txt")
    output_file_path_3 = os.path.join(uppaal_folder, "decentralized_stream_result_3.txt")
    verifyta_command = "verifyta-stratego-10"
    insert_paths_in_uppaal_model(model_template_path_1, weather_forecast_path,
                                 os.path.join(uppaal_folder, "libtable.so"))
    insert_paths_in_uppaal_model(model_template_path_2, weather_forecast_path,
                                 os.path.join(uppaal_folder, "libtable.so"))
    insert_paths_in_uppaal_model(model_template_path_3, weather_forecast_path,
                                 os.path.join(uppaal_folder, "libtable.so"))

    # Define uppaal model variables.
    action_variable = "Open"  # Name of the control variable
    debug = True  # Whether to run in debug mode.
    period = 60  # Control period in time units (minutes).
    horizon = 4 # How many periods to compute strategy for.
    uncertainty = 0.1  # The uncertainty in the weather forecast generation.

    # Get model and learning config dictionaries from files.

    with open(learning_config_path, "r") as yamlfile:
        learning_cfg_dict = yaml.safe_load(yamlfile)

    with open(model_config_path_1, "r") as yamlfile:
        model_cfg_dict_1 = yaml.safe_load(yamlfile)
    with open(model_config_path_2, "r") as yamlfile:
        model_cfg_dict_2 = yaml.safe_load(yamlfile)
    with open(model_config_path_3, "r") as yamlfile:
        model_cfg_dict_3 = yaml.safe_load(yamlfile)

    # Construct the MPC object.
    controllers = []
    #for i in range(num_basins):
    controller1 = MPCSetupPond(model_template_path_1, output_file_path_1, query_file=query_file_path_1,
                                model_cfg_dict=model_cfg_dict_1,
                                learning_args=learning_cfg_dict,
                                verifyta_command=verifyta_command,
                                external_simulator=False,
                                action_variable=action_variable, debug=debug)
    controllers.append(controller1)

    controller2 = MPCSetupPond(model_template_path_2, output_file_path_2, query_file=query_file_path_2,
                               model_cfg_dict=model_cfg_dict_2,
                               learning_args=learning_cfg_dict,
                               verifyta_command=verifyta_command,
                               external_simulator=False,
                               action_variable=action_variable, debug=debug)
    controllers.append(controller2)

    controller3 = MPCSetupPond(model_template_path_3, output_file_path_3, query_file=query_file_path_3,
                               model_cfg_dict=model_cfg_dict_3,
                               learning_args=learning_cfg_dict,
                               verifyta_command=verifyta_command,
                               external_simulator=False,
                               action_variable=action_variable, debug=debug)
    controllers.append(controller3)

    # try1. generating controllers with same model
    # try2. generating controllers with different model


    swmm_control(swmm_inputfile, orifice_id, basin_id, junction_id, time_step, swmm_results, controllers,
                 period, horizon, rain_data_file, weather_forecast_path, uncertainty)
    print("procedure completed!")


if __name__ == "__main__":
    main()
