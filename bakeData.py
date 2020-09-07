import sys
import os
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
                        dtype={'fips': object})  # preserve leading zero

# --- get county for each station ---
# use a dict for fast lookup
stationToCounty = stationDF['fips'].to_dict()

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

# --- gather average temp/precip by county ---

# create a dict mapping dates to dicts mapping
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
        countyPrecip = dateToPrecipDataByCounty[d][c]
        if countyPrecip:
            avgP = np.average(countyPrecip)
        else:
            avgP = np.NaN
        avgPrecip.append(avgP)
    i += 1
    sys.stdout.write(f"\r{i}/{len(dateToTempDataByCounty)} dates done")
    sys.stdout.flush()
print()  # newline after rewriting progress

# combine our lists of counties, dates, and avgs
# transpose to produce intended shape
countyData = np.array([counties, dates, avgTemps, avgPrecip]).transpose()
countyDF = pd.DataFrame(data=countyData,
                        columns=("fips", "date", "tavg", "prcp"))
countyDF = countyDF.astype({'tavg': int})
# sort by county, use stable sort to preserve date sorting
countyDF.sort_values('fips',
                     kind='mergesort',
                     inplace=True)

# --- get a rolling average of the last N days ---
# --- for temp/precip data by county ---

rollAvgTempWindow = 14
rollAvgPrecipWindow = 7
print("getting rolling averages. "
      f"Temperature roll window: {rollAvgTempWindow}"
      f"Precip roll window: {rollAvgPrecipWindow}")

# the data is sorted by county then date
# so the rolling average carries one county's data
# over to the next for the window of rolling (N).
# The first N days of a rolling avg should be NaN
# anyway so we allow this junk data and then replace
# with NaNs
rollAvgTempDF = countyDF['tavg'].rolling(rollAvgTempWindow).mean()
# find invalid dates by getting the first N dates
invalidRollDates = list(countyDF['date'].head(rollAvgTempWindow))
# create a mask that's true for every instance of those dates
rollMask = countyDF['date'].isin(invalidRollDates)
rollAvgTempDF = rollAvgTempDF.mask(rollMask)

countyDF['rtavg'] = rollAvgTempDF

rollAvgPrecipDF = countyDF['prcp'].rolling(rollAvgPrecipWindow).mean()
invalidRollDates = list(countyDF['date'].head(rollAvgPrecipWindow))
rollMask = countyDF['date'].isin(invalidRollDates)
rollAvgPrecipDF = rollAvgPrecipDF.mask(rollMask)
recentRain = [p > 0 if pd.notna(p) else np.NaN for p in rollAvgPrecipDF]

countyDF['rprcp'] = recentRain

# --- drop dates for which there are any temp NaNs ---

oldSize = len(countyDF)
countyDF.dropna(subset=['tavg', 'rtavg'], inplace=True)
print(f"dropped {oldSize - len(countyDF)} rows for NaN temps")

# --- bucket temp data ---
bucketSize = 5
print(f"bucketing temperature data. bucket size: {bucketSize} degrees")

# bucket by subtracting temp % bucketSize, giving the nearest
# int divisible by bucketSize
countyDF['tavg'] = countyDF['tavg'].apply(lambda x: x - x % bucketSize)
countyDF['rtavg'] = countyDF['rtavg'].apply(lambda x: x - x % bucketSize)


# join fire info


# write out DF
countyDF.to_csv("countyAvgs.csv", index=False)
