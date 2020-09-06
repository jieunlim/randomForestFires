import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

years = [2008, 2009, 2010, 2011, 2012, 2013]
weatherDFs = []
fireDFs = []

if not os.path.exists("countyAvgs.csv"):
    for y in years:
        weatherDFs.append(pd.read_csv(f"cleanedData/weather{y}.csv"))
        fireDFs.append(pd.read_csv(f"cleanedData/fireStats{y}.csv"))

    stationDF = pd.read_csv("cleanedData/geocodedStations.csv",
                            index_col=0,
                            dtype={'fips': object})  # preserve leading zero

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
    print()  # newline after rewriting progress


    # get rolling average temp/precip?


    # gather average temp/precip by county

    # create a dict mapping dates to a dict mapping i
    # counties to lists of data points

    defaultdictOfList = lambda: defaultdict(list)
    dateToTempDataByCounty = defaultdict(defaultdictOfList)
    dateToPrecipDataByCounty = defaultdict(defaultdictOfList)

    print("clustering weather data by county")
    i = 0

    for wdf in weatherDFs:
        for r in wdf.itertuples():
            if pd.notna(r.tavg):
                dateToTempDataByCounty[r.date][r.fips].append(r.tavg)
            if pd.notna(r.prcp):
                dateToPrecipDataByCounty[r.date][r.fips].append(r.prcp)
        i += 1
        sys.stdout.write(f"\r{i}/{len(weatherDFs)} years done")
        sys.stdout.flush()
    print()  # newline after rewriting progress


    # build a df where for every date we have avg data for each county
    dates = []
    counties = []
    avgTemps = []
    avgPrecip = []

    print("getting avg data by county")
    i = 0

    for d in dateToTempDataByCounty:
        for c in dateToTempDataByCounty[d]:
            dates.append(d)
            counties.append(c)
            avgT = int(np.average(dateToTempDataByCounty[d][c]))
            avgTemps.append(avgT)
            avgP = np.average(dateToPrecipDataByCounty[d][c])
            avgPrecip.append(avgP)
        i += 1
        sys.stdout.write(f"\r{i}/{len(dateToTempDataByCounty)} dates done")
        sys.stdout.flush()
    print()  # newline after rewriting progress

    countyData = np.array([counties, dates, avgTemps, avgPrecip]).transpose()
    countyDF = pd.DataFrame(data=countyData,
                            columns=("fips", "date", "tavg", "prcp"))
    # sort by county, use stable sort to preserve date sorting
    countyDF.sort_values('fips',
                         kind='mergesort',
                         inplace=True)

    countyDF.to_csv("countyAvgs.csv")
else:
    countyDF = pd.read_csv("countyAvgs.csv")
    #countyDF.drop('rtavg', axis=1)
    print("using cached countyAvgs")

# get a rolling average of the last N days
# for temp/precip data by county

# the data is sorted by county then date
# so the rolling average carries one county's data
# over to the next for the window of rolling (N).
# The first N days of a rolling avg should be NaN
# anyway so we allow this junk data and then replace
# with NaNs
rollAvgTempWindow = 14
rollAvgTempDF = countyDF['tavg'].rolling(rollAvgTempWindow).mean()
invalidRollDates = countyDF['date'].isin(countyDF['date'].head(rollAvgTempWindow))
rollAvgTempDF.mask(invalidRollDates, '')

countyDF['rtavg'] = rollAvgTempDF

rollAvgPrecipWindow = 7
rollAvgPrecipDF = countyDF['prcp'].rolling(rollAvgPrecipWindow).mean()
invalidRollDates = countyDF['date'].isin(countyDF['date'].head(rollAvgPrecipWindow))
rollAvgPrecipDF.mask(invalidRollDates, '')
recentRain = [p > 0 if p != np.NaN else np.NaN for p in rollAvgPrecipDF]

countyDF['rprecip'] = recentRain

countyDF.to_csv("countyAvgs.csv")

# bucket temp by county


# join fire info


# write out DF
