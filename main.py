import os
import xml.etree.ElementTree as ET
import csv
import re
import strategoutil as sutil
import datetime
import sys
import yaml
import inspect

from pyswmm import Simulation, Nodes, Links



def swmm_control(swmm_inputfile, orifice_id, basin_id, time_step, csv_file_basename, controllers,
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

    with Simulation(swmm_inputfile) as sim:
        sys.stdout.write('\n')
        i = 0
        interval = sim.end_time - sim.start_time
        duration = interval.total_seconds() / 3600
        print_progress_bar(i, duration, "progress")
        sus = []
        orifices = []
        for j in range(len(controllers)):
            sus.append(Nodes(sim)[f"{basin_id}{j+1}"])
            orifices.append(Links(sim)[f"{orifice_id}{j+1}"])
            orifice_settings.append([])
            water_depth_basins.append([])
        ca = Subcatchments(sim)["S1"]
        sim.step_advance(time_step)
        current_time = sim.start_time

        for controller, orifice, orifice_setting, su, water_depth_basin in \
                zip(controllers, orifices, orifice_settings, sus, water_depth_basins):
            orifice.target_setting = get_control_strategy(su.depth, current_time, controller,
                                                          period, horizon, rain_data_file,
                                                          weather_forecast_path, uncertainty)
            orifice_setting.append(1.75*orifice.target_setting + 2)  # Offset for printing purposes
            water_depth_basin.append(su.depth)
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
                # Set the control parameter
                orifice.target_setting = get_control_strategy(su.depth, current_time, controller,
                                                              period, horizon, rain_data_file,
                                                              weather_forecast_path, uncertainty)
                orifice_setting.append(1.75*orifice.target_setting + 2)  # Offset for printing purposes

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
            writer.writerow(line)





def insert_paths_in_uppaal_model(uppaal_model, weather_forecast_path, libtable_path):
    """
    Insert the provided weather forecast path into the uppaal model.

    :param str uppaal_model: uppaal model path
    :param str weather_forecast_path: weather forecast path
    :param str libtable_path: libtable.so path
    """
    with open(uppaal_model, "r+") as f:
        file_content = f.read()
        # insert path for model RAIN's decralation.
        new_line = "const int file_id = table_read_csv(\"" + weather_forecast_path + "\""
        file_content = re.sub(r"const int file_id = table_read_csv\(\"[^\"]*\"", new_line,
                              file_content, count=1) # "count" limits max number of matches.

        # insert path for global declaration.
        new_line = "import \"" + libtable_path + "\""
        file_content = re.sub(r"import \"[^\"]*\"", new_line, file_content, count=1)

        f.seek(0) ### no idea ...
        f.write(file_content)
        f.truncate() ### no idea ...

def insert_paths_in_test():
    with open("/home/esther/Downloads/DIREC/Compositionality/test.txt", mode="a") as file:
        words = ["Python\n", "YUNDAEHEE\n", "076923\n"]

        file.write("START\n")
        file.writelines(words)
        file.write("END")
        file.truncate()

class MPCSetupPond(sutil.SafeMPCSetup):
    def create_query_file(self, horizon, period, final): ### no idea .. where does the parameters come from?
        """
        Create the query file for each step of the pond model.
        Current content will be overwritten.

        Overrides SafeMPCsetup.create_query_file().
        """
        with open(self.query_file, "w") as f:
            line1 = f"strategy opt = minE (c) [<={horizon}*{period}]: <> (t=={final})\n" # where does horizon, period, final come from?
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


def main():
    # First, figure out where this script is located. This is OS dependent.
    this_file = os.path.realpath(__file__)
    base_folder = os.path.dirname(this_file)

    swmm_folder = "swmm"  # swmm model located folder name
    swmm_inputfile = os.path.join(base_folder, swmm_folder,"swmm.inp")  # CLAIRE/swmm_models/swmm_decentralized.inp
    assert (os.path.isfile(swmm_inputfile))  # if there is the file, go ahead!

    # We found the model. Now we have to include the correct path to the rain data into the model.
    rain_data_file = "swmm_5061.dat"  # Assumed to be in the same folder as the swmm model input file.
    rain_data_file = os.path.join(base_folder, swmm_folder, rain_data_file)


    # Second, setup path for uppaal model.
    uppaal_folder_name = "uppaal"
    uppaal_folder = os.path.join(base_folder, uppaal_folder_name)
    model_template_path = os.path.join(uppaal_folder, "pond.xml")
    query_file_path = os.path.join(uppaal_folder, "pond.q")
    model_config_path = os.path.join(uppaal_folder, "pond.yaml")
    learning_config_path = os.path.join(uppaal_folder, "verifyta_pond_config.yaml")
    weather_forecast_path = os.path.join(uppaal_folder, "weather_forecast.csv")
    output_file_path = os.path.join(uppaal_folder, "pond_result.txt")
    verifyta_command = "verifyta-stratego-10"
    insert_paths_in_uppaal_model(model_template_path, weather_forecast_path,
                                 os.path.join(uppaal_folder, "libtable.so"))
    #insert_paths_in_test()

    # Third, define uppaal model variables. It is for Online synthesis.
    action_variable = "Open"  # Name of the control variable
    debug = True  # Whether to run in debug mode.
    period = 60  # Control period in time units (minutes).
    horizon = 2  # How many periods to compute strategy for.
    uncertainty = 0.1  # The uncertainty in the weather forecast generation.

    # Fourth, get model and learning config dictionaries from files.
    with open(model_config_path, "r") as yamlfile:
        model_cfg_dict = yaml.safe_load(yamlfile)
    with open(learning_config_path, "r") as yamlfile:
        learning_cfg_dict = yaml.safe_load(yamlfile)

    # Fifth, Construct the MPC object.
        num_basins = 1

        controllers = []
        for i in range(num_basins):
            controller = MPCSetupPond(model_template_path, output_file_path, query_file=query_file_path,
                                      model_cfg_dict=model_cfg_dict,
                                      learning_args=learning_cfg_dict,
                                      verifyta_command=verifyta_command,
                                      external_simulator=False,
                                      action_variable=action_variable, debug=debug)
            controllers.append(controller)
            #print(controllers)

            #print(inspect.getfile(sutil))

            swmm_control(swmm_inputfile, orifice_id, basin_id, time_step, swmm_results, controllers, period, horizon,
                         rain_data_file, weather_forecast_path, uncertainty)
            print("procedure completed!")




if __name__ == "__main__":
    main()
