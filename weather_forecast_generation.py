import csv
import datetime
from functools import cache


def create_weather_forecast(rain_data_file, weather_forecast_file, start_date, horizon,
                            uncertainty):
    """
    Create a weather forecast from historical rain data.
    """
    rain_data = read_rain_data(rain_data_file)
    weather_forecast = calculate_weather_intervals(rain_data, start_date, horizon, uncertainty)
    write_weather_forecast(weather_forecast_file, weather_forecast)


@cache
def read_rain_data(data_file):
    """
    Read the historical rain data from file.
    """
    with open(data_file, "r") as f:
        lines = f.readlines()
        rain_data = []
        for line in lines:
            words = line.split()
            # In <name>.dat rain file, we have mm/dd/yyyy hh:mm:ss x.y per line.
            dt = datetime.datetime.strptime(words[0] + " " + words[1], "%m/%d/%Y %H:%M:%S")
            rain_data.append([dt, float(words[2]) / 60])  # Divided by 60 to get mm/h -> mm/min.
    return rain_data


def calculate_weather_intervals(rain_data, start_date, horizon, uncertainty):
    """
    Calculate lower and upper bounds of dry and rain intervals starting from a specified date and
    time. The horizon for creating these intervals are assumed to be given in minutes.
    """
    # Check start_date.
    assert horizon > 0
    assert start_date >= rain_data[0][0]
    assert start_date + datetime.timedelta(minutes=horizon) <= rain_data[len(rain_data) - 1][0]

    # Go in rain_data to start date and time. Remember that each line represents 1 minute.
    index = 0
    while rain_data[index][0] != start_date:
        index += 1

    # Initialize variables.
    raining = rain_data[index][1] > 0
    start_current_interval = index
    cumulative_rain = 0.0
    weather_intervals = []
    next_interval = []

    # If we start with rain, the first dry interval will be dummy values.
    if raining:
        next_interval.append(0)
        next_interval.append(0)

    while rain_data[index][0] <= start_date + datetime.timedelta(minutes=horizon):
        current_rain = rain_data[index][1]
        if raining and current_rain > 0:
            # It was raining and it is still raining.
            cumulative_rain += current_rain

        elif raining and current_rain == 0:
            # It was raining but it stopped raining.
            interval_duration = index - start_current_interval
            next_interval.append(int(round(interval_duration * (1 - uncertainty))))
            next_interval.append(int(round(interval_duration * (1 + uncertainty))))
            start_current_interval = index
            next_interval.append(cumulative_rain / interval_duration)
            weather_intervals.append(next_interval)
            next_interval = []
            raining = False

        elif not raining and current_rain == 0:
            # It was dry and it is still dry. Nothing special to do.
            pass

        elif not raining and current_rain > 0:
            # It was dry but it started raining.
            interval_duration = index - start_current_interval
            next_interval.append(int(round(interval_duration * (1 - uncertainty))))
            next_interval.append(int(round(interval_duration * (1 + uncertainty))))
            start_current_interval = index
            cumulative_rain = current_rain
            raining = True

        else:
            # We should not be here.
            raise RuntimeError("Rain data has unexpected values.")

        index += 1

    # Wrap up the last next_interval array.
    interval_duration = index - start_current_interval
    if interval_duration == horizon + 1:  # Nothing will change.
        next_interval.append(interval_duration)
        next_interval.append(interval_duration)
    else:
        next_interval.append(int(round(interval_duration * (1 - uncertainty))))
        next_interval.append(int(round(interval_duration * (1 + uncertainty))))

    if raining:
        next_interval.append(cumulative_rain / interval_duration)
    else:
        next_interval.append(0)
        next_interval.append(0)
        next_interval.append(0)
    weather_intervals.append(next_interval)

    # Add two days of dry weather for safety in Uppaal Stratego.
    next_interval = [24 * 60, 24 * 60, 24 * 60, 24 * 60, 0]
    weather_intervals.append(next_interval)
    return weather_intervals


def write_weather_forecast(file, weather_forecast):
    """
    Writes the weather forecast to a csv file. Overwrites current content.
    """
    with open(file, "w") as f:
        weather_writer = csv.writer(f, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        weather_writer.writerow(['#dryL', 'dryU', 'rainL', 'rainU', 'rain'])
        for row in weather_forecast:
            weather_writer.writerow(row)


if __name__ == "__main__":
    pass
