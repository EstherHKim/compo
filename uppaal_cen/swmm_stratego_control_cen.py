from pyswmm import Simulation, Nodes, Links, RainGages, Subcatchments
import os.path
import csv
import re
import strategoutil as sutil
import datetime
import sys
import yaml
import weather_forecast_generation as weather


def swmm_control(swmm_inputfile, orifice_id, basin_id, time_step, csv_file_basename, controller, num_basins,
                 period, horizon, rain_data_file, weather_forecast_path, uncertainty):
    """
    Implement the control strategy from uppal stratego to swmm
    requires:
        swmm_inputfile: path
        orifice_id: string
        open_settings: percentage of orifice opening, float, from 0 to 1
        basin_id: string, upstream basin ID
        time_step: float
        csv_file_basename: path

    returns:
        one csv files with three columns:
        col 0. time 
        col 1. the upstream pond water level changes when implemented control strategy, meter above the bottom elevation of basin
        col 2. the orifice flow (discharge) changes when implemented control strategy, cubic meter per second
    """
    time_series = []
    water_depth_basins = []
    water_depth_junctions = [] # for stream
    orifice_settings = []
    rain = []
    weather_forecast_low = []
    weather_forecast_high = []
    weather_forecast_int = []



    # subcatchment_total_rain = []
    # subcatchment_total_runoff = []
    # subcatchment_total_infiltration = []
    # subcatchment_total_evaporation = []

    with Simulation(swmm_inputfile) as sim:
        sys.stdout.write('\n')
        i = 0
        interval = sim.end_time - sim.start_time
        duration = interval.total_seconds() / 3600
        print_progress_bar(i, duration, "progress")
        sus = []
        orifices = []
        junctions = []  # for stream

        for j in range(num_basins):
            sus.append(Nodes(sim)[f"{basin_id}{j+1}"])
            orifices.append(Links(sim)[f"{orifice_id}{j+1}"])
            water_depth_basins.append([])
            orifice_settings.append([])
            junctions.append(Nodes(sim)[f"{junction_id}{j+2}"])
            water_depth_junctions.append([])

        ca = Subcatchments(sim)["S1"]
        sim.step_advance(time_step)
        current_time = sim.start_time

        # important!
        target_settings = get_control_strategy(sus, current_time, controller, period,
                                                      horizon, rain_data_file,
                                                      weather_forecast_path, uncertainty)

        for j in range(num_basins):
            orifices[j].target_setting = target_settings[j]
            #orifice_settings[j].append(1.75 * target_settings[j] + 2)
            orifice_settings[j].append(target_settings[j] )
            water_depth_basins[j].append(sus[j].depth)
            water_depth_junctions[j].append(sus[j].depth)

        rain_low, rain_high, rain_int = get_weather_forecast_result(weather_forecast_path)
        weather_forecast_low.append(rain_low)
        weather_forecast_high.append(rain_high)
        weather_forecast_int.append(rain_int)
        time_series.append(sim.start_time)
        rain.append(0)
        total_precipitation = 0

        total_inflow = 0
        total_outflow = 0
        total_rainfall = 0

        for step in sim:
            current_time = sim.current_time
            time_series.append(current_time)
            # water_depth_basin1.append(su1.depth)
            # water_depth_basin2.append(su2.depth)

            rain.append(ca.statistics.get('precipitation') - total_precipitation)
            total_precipitation = ca.statistics.get('precipitation')

            i = i + 1
            print_progress_bar(i, duration, "progress")

            # Set the control parameter
            # important!
            target_settings = get_control_strategy(sus, current_time, controller, period,
                                                   horizon, rain_data_file,
                                                   weather_forecast_path, uncertainty)
            #orifice.target_setting = get_control_strategy(su1.depth, current_time, controller,
                                                          #period, horizon, rain_data_file,
                                                          #weather_forecast_path, uncertainty)

            # orifice_settings.append(1.75*orifice.target_setting + 2)
            orifice_settings.append(orifice.target_setting)

            rain_low, rain_high, rain_int = get_weather_forecast_result(weather_forecast_path)
            weather_forecast_low.append(rain_low)
            weather_forecast_high.append(rain_high)
            weather_forecast_int.append(rain_int)

            # orifice_flow.append(orifice.flow)
            # rain.append(rg1.rainfall)
            # total_inflow = total_inflow + su.total_inflow
            # basin_total_inflow.append(total_inflow)
            # overflow.append(su.statistics.get('flooding_volume'))
            # total_outflow = total_outflow + su.total_outflow
            # basin_total_outflow.append(total_outflow)
            # basin_total_evaporation.append(su.storage_statistics.get('evap_loss'))
            # total_rainfall = total_rainfall + ca.rainfall
            # subcatchment_total_rain.append(total_rainfall)
            # subcatchment_total_runoff.append(ca.statistics.get('runoff'))
            # subcatchment_total_infiltration.append(ca.statistics.get('infiltration'))
            # subcatchment_total_evaporation.append(ca.statistics.get('evaporation'))

    i = i + 1
    print_progress_bar(i, duration, "progress")
    dirname = os.path.dirname(swmm_inputfile)
    output_csv_file = os.path.join(dirname, csv_file_basename + "." + "csv")
    with open(output_csv_file, "w") as f:
        writer = csv.writer(f)
        top_row = ["time", "rain", "forecast_low", "forecast_high", "forecast_int"]
        for j in range(len(controllers)):
            top_row.extend([f"water_depth_basin{j + 1}"])
        for j in range(len(controllers)):
            top_row.extend([f"orifice_setting{j + 1}"])
        top_row.extend([f"water_depth_junction{j + 2}"])
        writer.writerow(top_row)


       # writer.writerow(["time", "water_depth_basin1", "water_depth_basin2", "orifice_setting",
       #                  "rain", "forecast_low", "forecast_high", "forecast_int"])
       # for i, j, k, l, m, n, o, p in zip(time_series, water_depth_basin1, water_depth_basin2,
       #                                  orifice_settings, rain, weather_forecast_low,
       #                                   weather_forecast_high, weather_forecast_int):
       #    i = i.strftime('%Y-%m-%d %H:%M')
       #    writer.writerow([i, j, k, l, m, n, o, p])

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
            line.append(water_depth_junction[line_index])
            writer.writerow(line)

def get_control_strategy(sus, current_time, controller, period, horizon,
                         rain_data_file, weather_forecast_path, uncertainty):
    for j in range(len(sus)):
        controller.controller.update_state({f'w{j+1}': sus[j].depth * 100}) #  Conversion from m to cm.
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


class MPCSetupPond(sutil.SafeMPCSetup):
    def create_query_file(self, horizon, period, final):
        """
        Create the query file for each step of the pond model. Current
        content will be overwritten.

        Overrides SafeMPCsetup.create_query_file().
        """
        with open(self.queryfile, "w") as f:
            line1 = f"strategy opt = minE (st_c + c1 + c2+ c3) [<={horizon}*{period}]: <> (t=={final} && o <= 0)\n"
            f.write(line1)
            f.write("\n")
            line2 = f"simulate 1 [<={period}+1] {{ {self.controller.get_var_names_as_string()} " \
                    f"}} under opt\n"
            f.write(line2)

    def create_alternative_query_file(self, horizon, period, final):
        """
        Create an alternative query file in case the original query could not be satisfied by
        Stratego, i.e., it could not find a strategy. Current content will be overwritten.

        Overrides SafeMPCsetup.create_alternative_query_file().
        """
        with open(self.queryfile, "w") as f:
            line1 = f"strategy opt = minE (wmax) [<={horizon}*{period}]: <> (t=={final})\n"
            f.write(line1)
            f.write("\n")
            line2 = f"simulate 1 [<={period}+1] {{ {self.controller.get_var_names_as_string()} " \
                    f"}} under opt\n"
            f.write(line2)

    def perform_at_start_iteration(self, controlperiod, horizon, duration, step, **kwargs):
        """
        Performs some customizable preprocessing steps at the start of each MPC iteration.

        Overrides SafeMPCsetup.perform_at_start_iteration().
        """
        current_date = kwargs["start_date"] + datetime.timedelta(hours=step)
        weather.create_weather_forecast(kwargs["historical_rain_data_path"],
                                        kwargs["weather_forecast_path"], current_date,
                                        horizon * controlperiod, kwargs["uncertainty"])


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


def main():
    this_file = os.path.realpath(__file__)
    base_folder = os.path.dirname(os.path.dirname(this_file))

    # Setting directory for swmm.
    swmm_folder = "swmm"  # swmm model located folder name
    swmm_inputfile = os.path.join(base_folder, swmm_folder, "swmm_centralized.inp")  # CLAIRE/swmm_models/swmm_decentralized.inp
    assert (os.path.isfile(swmm_inputfile))  # if there is the file, go ahead!

    # We found the model. Now we have to include the correct path to the rain data into the model.
    rain_data_file = "swmm_5061.dat"  # Assumed to be in the same folder as the swmm model input file.
    rain_data_file = os.path.join(base_folder, swmm_folder, rain_data_file)
    insert_rain_data_file_path(swmm_inputfile, rain_data_file)

    # Finally we can specify other variables of swimm.
    orifice_id = "OR"
    basin_id = "SU"
    junction_id = "J"  # for stream
    num_basins = 3  # in .inp file, only 3 ponds exits.
    time_step = 60 * 60  # 60 seconds/min x 60 min/h -> 1 h
    swmm_results = "swmm_centralized_results"

    # Now we locate the Uppaal folder and files.
    uppaal_folder_name = "uppaal_cen"
    uppaal_folder = os.path.join(base_folder, uppaal_folder_name)
    model_template_path = os.path.join(uppaal_folder, "pond_centralized_stream.xml")
    query_file_path = os.path.join(uppaal_folder, "pond_centralized_stream_query.q")
    model_config_path = os.path.join(uppaal_folder, "pond_centralized_stream_config.yaml")
    learning_config_path = os.path.join(uppaal_folder, "verifyta_centralized_stream_config.yaml")
    weather_forecast_path = os.path.join(uppaal_folder, "centralized_weather_forecast.csv")
    output_file_path = os.path.join(uppaal_folder, "centralized_stream_result.txt")
    verifyta_command = "verifyta-stratego-10"
    insert_paths_in_uppaal_model(model_template_path, weather_forecast_path,
                                 os.path.join(uppaal_folder, "libtable.so"))

    # Define uppaal model variables.
    action_variable = "Open"  # Name of the control variable.
    debug = True  # Whether to run in debug mode.
    period = 60  # Control period in time units (minutes).
    horizon = 4  # How many periods to compute strategy for.
    uncertainty = 0.1  # The uncertainty in the weather forecast generation.

    # Get model and learning config dictionaries from files.
    with open(model_config_path, "r") as yamlfile:
        model_cfg_dict = yaml.safe_load(yamlfile)
    with open(learning_config_path, "r") as yamlfile:
        learning_cfg_dict = yaml.safe_load(yamlfile)

    # Construct the MPC object.
    # one controller for the system composed of 3 ponds.
    controller = MPCSetupPond(model_template_path, output_file_path, queryfile=query_file_path,
                              model_cfg_dict=model_cfg_dict,
                              learning_args=learning_cfg_dict,
                              verifyta_command=verifyta_command,
                              external_simulator=False,
                              action_variable=action_variable, debug=debug)

    swmm_control(swmm_inputfile, orifice_id, basin_id, time_step, swmm_results, controller, num_basins,
                 period, horizon, rain_data_file, weather_forecast_path, uncertainty)
    print("procedure completed!")


if __name__ == "__main__":
    main()
