#! /bin/bash
if [ -z "$1" ]
then
    version=$(git rev-parse HEAD)
else
    version=$1
fi

locations="atlanta seattle hyderabad mumbai taipei appleton dharamshala fremont"


archive () {
    flytectl -c .flyte/remote-config.yaml \
        update launchplan \
        -p flytelab \
        -d development \
        "$1_weather_forecast_v2" \
        --version $version \
        --archive
}

for location in $locations
do
    echo archiving launch plan version $version for $location
    archive $location
done
