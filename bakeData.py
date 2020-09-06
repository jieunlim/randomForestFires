import sys
import pandas as pd
import numpy as np
from collections import defaultdict

years = [2008, 2009, 2010, 2011, 2012, 2013]
weatherDFs = []
fireDFs = []

for y in years:
    weatherDFs.append(pd.read_csv(f"cleanedData/weather{y}.csv"))
    fireDFs.append(pd.read_csv(f"cleanedData/fireStats{y}.csv"))

stationDF = pd.read_csv("cleanedData/geocodedStations.csv", 
                        index_col=0, 
                        dtype={'fips':object}) # preserve leading zero

# get county for each station
stationToCounty = stationDF['fips']

print("attributing counties to station")

i = 0
for wdf in weatherDFs:
    stationCountyList = []
    for r in wdf.itertuples():
        try:
            c = stationToCounty[r.station]
        except KeyError:
            c = np.NaN
        stationCountyList.append(c)
    wdf['fips'] = stationCountyList
    i += 1
    sys.stdout.write(f"\r{i}/{len(weatherDFs)} years done")
    sys.stdout.flush()


# get rolling average temp/precip?


# average temp/precip by county


# bucket temp by county


# join fire info


# write out DF
