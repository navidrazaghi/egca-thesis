"""Weather presets: 14 train combinations + held-out OOD test conditions
(Sec. 5-1 / 5-4-1). Values map to carla.WeatherParameters fields.
"""
TRAIN_WEATHERS = {
    "ClearNoon": dict(cloudiness=15, precipitation=0, fog_density=0,
                      sun_altitude_angle=75),
    "ClearSunset": dict(cloudiness=15, precipitation=0, fog_density=0,
                        sun_altitude_angle=15),
    "CloudyNoon": dict(cloudiness=80, precipitation=0, fog_density=0,
                       sun_altitude_angle=75),
    "CloudySunset": dict(cloudiness=80, precipitation=0, fog_density=0,
                         sun_altitude_angle=15),
    "WetNoon": dict(cloudiness=20, precipitation=0, precipitation_deposits=50,
                    fog_density=0, sun_altitude_angle=75),
    "WetSunset": dict(cloudiness=20, precipitation=0, precipitation_deposits=50,
                      fog_density=0, sun_altitude_angle=15),
    "MidRainyNoon": dict(cloudiness=80, precipitation=30,
                         precipitation_deposits=40, fog_density=5,
                         sun_altitude_angle=75),
    "MidRainSunset": dict(cloudiness=80, precipitation=30,
                          precipitation_deposits=40, fog_density=5,
                          sun_altitude_angle=15),
    "HardRainNoon": dict(cloudiness=90, precipitation=80,
                         precipitation_deposits=80, fog_density=7,
                         sun_altitude_angle=75),
    "SoftRainNoon": dict(cloudiness=70, precipitation=15,
                         precipitation_deposits=30, fog_density=3,
                         sun_altitude_angle=75),
    "FogMorning": dict(cloudiness=40, precipitation=0, fog_density=60,
                       fog_distance=8, sun_altitude_angle=25),
    "ClearNight": dict(cloudiness=10, precipitation=0, fog_density=0,
                       sun_altitude_angle=-80),
    "CloudyNight": dict(cloudiness=80, precipitation=0, fog_density=0,
                        sun_altitude_angle=-80),
    "WetNight": dict(cloudiness=40, precipitation=0, precipitation_deposits=60,
                     fog_density=3, sun_altitude_angle=-80),
}

# held-out, out-of-distribution (never in the training set)
TEST_WEATHERS = {
    "HardRainNight": dict(cloudiness=95, precipitation=90,
                          precipitation_deposits=90, fog_density=10,
                          sun_altitude_angle=-80),
    "HeavyFogNoon": dict(cloudiness=60, precipitation=0, fog_density=85,
                         fog_distance=5, sun_altitude_angle=75),
}

# the six evaluation conditions of Table 5-4
EVAL_CONDITIONS = ["ClearNoon", "WetNoon", "HardRainNoon", "FogMorning",
                   "ClearNight", "HardRainNight"]


def apply_weather(world, name):
    import carla
    params = {**TRAIN_WEATHERS, **TEST_WEATHERS}[name]
    w = carla.WeatherParameters()
    for k, v in params.items():
        setattr(w, k, float(v))
    world.set_weather(w)
