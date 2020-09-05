#! /usr/bin/python3

import requests
import sys
import pandas as pd

# import station data, which includes lat/long
stations = pd.read_csv("./rawData/allStations.csv", index_col=0)
numStations = len(stations)

county = []
fips = []

# the fcc provides an api for getting location data by lat/long
baseURL = "https://geo.fcc.gov/api/census/area?lat={}&lon={}&format=json"

i = 0
for s in stations.itertuples():
    # query the fcc api for the station's lat/long
    r = requests.get(baseURL.format(s.LATITUDE, s.LONGITUDE))
    data = r.json()["results"][0]

    # get county and fips data
    county.append(data["county_name"])
    fips.append(data["county_fips"])
    i += 1
    sys.stdout.write(f"\r{i}/{numStations} stations gathered")
    sys.stdout.flush()

# add that data to our DF
stations["county"] = county
stations["fips"] = fips

# output our geocoded data
stations.to_csv("cleanedData/geocodedStations.csv")

print("")
print(stations)
