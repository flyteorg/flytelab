from setuptools import setup, find_packages

setup(
    name="flytelab-weather-forecasting",
    version="0.0.0+dev0",
    description="Weather forecasting application using Flytekit.",
    author="flyteorg",
    author_email="admin@flyte.org",
    namespace_packages=["flytelab"],
    packages=["flytelab.weather_forecasting"],
)
