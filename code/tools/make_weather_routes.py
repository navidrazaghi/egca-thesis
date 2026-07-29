"""Write a copy of the Longest6 route file with one fixed weather preset.

Longest6 ships a weather condition baked into every `<route>` element, chosen
per route.  That is the right thing for the headline benchmark number, but it
makes the weather-robustness study of Table 5-4 impossible: the condition has to
be the independent variable, identical across all 36 routes, or the comparison
between conditions also compares different routes.

The values written here come from `egca.carla_sim.weather`, the same table the
data collection used, so an evaluation in "HardRainNoon" sees exactly the
weather the training set called "HardRainNoon".  Attributes the preset does not
mention are written as 0.0, which is what `carla.WeatherParameters()` defaults
to and therefore what `apply_weather` leaves them at during collection.

Usage:
    python tools/make_weather_routes.py --routes $ROUTES6/longest6.xml \
        --weather HardRainNoon --out routes_weather/longest6_HardRainNoon.xml

    # or all six conditions of Table 5-4 at once
    python tools/make_weather_routes.py --routes $ROUTES6/longest6.xml \
        --all --outdir routes_weather
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from egca.carla_sim.weather import (  # noqa: E402
    EVAL_CONDITIONS, TEST_WEATHERS, TRAIN_WEATHERS,
)

PRESETS = {**TRAIN_WEATHERS, **TEST_WEATHERS}

# every attribute the leaderboard's route parser reads off a <weather> element
WEATHER_ATTRS = ["cloudiness", "precipitation", "precipitation_deposits",
                 "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle",
                 "fog_density", "fog_distance", "fog_falloff", "wetness"]


def rewrite(routes_xml, name):
    """Return an ElementTree of `routes_xml` with every route's weather set."""
    preset = PRESETS[name]
    tree = ET.parse(routes_xml)
    n_set, n_added = 0, 0
    for route in tree.getroot().iter("route"):
        weather = route.find("weather")
        if weather is None:
            weather = ET.SubElement(route, "weather")
            n_added += 1
        else:
            n_set += 1
        weather.set("id", name)
        for attr in WEATHER_ATTRS:
            weather.set(attr, "%.6f" % float(preset.get(attr, 0.0)))
    return tree, n_set, n_added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True, help="source longest6.xml")
    ap.add_argument("--weather", help="preset name from egca.carla_sim.weather")
    ap.add_argument("--all", action="store_true",
                    help="write every condition in EVAL_CONDITIONS")
    ap.add_argument("--out", help="output file (single --weather)")
    ap.add_argument("--outdir", default="routes_weather")
    args = ap.parse_args()

    if not args.all and not args.weather:
        ap.error("give --weather NAME or --all")
    names = EVAL_CONDITIONS if args.all else [args.weather]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        ap.error("unknown weather %s; known: %s"
                 % (", ".join(unknown), ", ".join(sorted(PRESETS))))

    for name in names:
        tree, n_set, n_added = rewrite(args.routes, name)
        out = args.out if (args.out and not args.all) else os.path.join(
            args.outdir, "longest6_%s.xml" % name)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        tree.write(out, encoding="utf-8", xml_declaration=True)
        print("%-16s -> %s  (%d routes rewritten, %d without a weather tag)"
              % (name, out, n_set, n_added))


if __name__ == "__main__":
    main()
