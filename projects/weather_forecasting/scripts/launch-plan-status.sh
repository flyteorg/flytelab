#! /bin/bash
if [ -z "$1" ]
then
    version=$(git rev-parse HEAD)
else
    version=$1
fi

locations="atlanta seattle hyderabad mumbai taipei appleton dharamshala"


get-status () {
    flytectl -c ~/.flyte/nuclydedemo-config.yaml \
        get launchplan \
        -p flytelab \
        -d development \
        -o yaml \
        --latest \
        "$1_weather_forecast_v2"
}

for location in $locations
do
    echo "launch plan status for $location, version: $version"
    get-status $location | grep state
    echo
done
