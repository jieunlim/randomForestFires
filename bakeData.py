import sys
import datetime as dt
import pandas as pd
import numpy as np
from collections import defaultdict

years = [2008, 2009, 2010, 2011, 2012, 2013]
weatherDFs = []
fireDFs = []
OUTPUT_FILE = "bakedData/fireWeatherData.csv"

for y in years:
    weatherDFs.append(pd.read_csv(f"cleanedData/weather{y}.csv"))
    fireDFs.append(pd.read_csv(f"cleanedData/fireStats{y}.csv"))

stationDF = pd.read_csv("cleanedData/geocodedStations.csv",
                        index_col=0,
                        dtype={'fips': object})  # preserve leading zero

# --- get county for each station ---

# lowercase everything for simpler string comparison
stationDF['county'] = stationDF['county'].apply(str.lower)
# use a dict for fast lookup
stationToFips = stationDF['fips'].to_dict()
fipsToCounty = stationDF[['fips', 'county']].set_index('fips').to_dict()['county']

# some stations only provide min/max temperature, we want average
# fill with the average of min/max, which should be a good approximation
print("filling in average temp where missing")
filledAvgs = 0
for wdf in weatherDFs:
    mask = wdf['tavg'].isna()
    wdf.loc[mask, 'tavg'] = (wdf['tmax'][mask] + wdf['tmin'][mask]) / 2
    filledAvgs += sum(mask)
print(f"filled averages for {filledAvgs} rows")

print("attributing counties to station")

i = 0
droppedRows = 0
for wdf in weatherDFs:
    stationCountyList = []
    for row in wdf.itertuples():
        try:
            f = stationToFips[row.station]
        except KeyError:
            f = np.NaN
        stationCountyList.append(f)
    wdf['fips'] = stationCountyList
    i += 1
    sys.stdout.write(f"\r{i}/{len(weatherDFs)} years done")
    # drop rows for which we have no fips
    oldsize = len(wdf)
    wdf.dropna(subset=['fips'], inplace=True)
    droppedRows += oldsize - len(wdf)
    sys.stdout.write(f"\t{droppedRows} rows dropped for bad location")
    sys.stdout.flush()
print()  # newline after rewriting progress

# --- gather average temp/precip by county ---

# create a dict mapping dates to dicts mapping
# counties to lists of data points
defaultdictOfList = lambda: defaultdict(list)
dateToTempDataByFips = defaultdict(defaultdictOfList)
dateToPrecipDataByCounty = defaultdict(defaultdictOfList)

print("clustering weather data by county")
i = 0

for wdf in weatherDFs:
    for row in wdf.itertuples():
        if pd.notna(row.tavg):
            dateToTempDataByFips[row.date][row.fips].append(row.tavg)
        if pd.notna(row.prcp):
            dateToPrecipDataByCounty[row.date][row.fips].append(row.prcp)
    i += 1
    sys.stdout.write(f"\r{i}/{len(weatherDFs)} years done")
    sys.stdout.flush()
print()  # newline after rewriting progress

# build a df where for every date we have avg data for each county
dates = []
fips = []
counties = []
avgTemps = []
avgPrecip = []

print("getting avg data by county")
i = 0
for d in dateToTempDataByFips:
    for f in dateToTempDataByFips[d]:
        dates.append(d)
        fips.append(f)
        try:
            counties.append(fipsToCounty[f])
        except KeyError:
            counties.append("UNKNOWN")
        avgT = int(np.average(dateToTempDataByFips[d][f]))
        avgTemps.append(avgT)
        countyPrecip = dateToPrecipDataByCounty[d][f]
        if countyPrecip:
            avgP = np.average(countyPrecip)
        else:
            avgP = np.NaN
        avgPrecip.append(avgP)
    i += 1
    sys.stdout.write(f"\r{i}/{len(dateToTempDataByFips)} dates done")
    sys.stdout.flush()
print()  # newline after rewriting progress

# combine our lists of counties, dates, and avgs
# transpose to produce intended shape
countyData = np.array([fips, counties, dates, avgTemps, avgPrecip]).transpose()
fireWeatherDF = pd.DataFrame(data=countyData,
                             columns=["fips", "county", "date", "tavg", "prcp"])
# force typing of some fields
fireWeatherDF = fireWeatherDF.astype({'tavg': int, 'prcp': float})
dtCols = fireWeatherDF['date'].apply(dt.datetime.fromisoformat)
fireWeatherDF['date'] = dtCols
# it looks like some stations send NaN instead of zero for no precip
fireWeatherDF['prcp'].fillna(0, inplace=True)
# sort by county, use stable sort to preserve date sorting
fireWeatherDF.sort_values(['fips', 'date'], inplace=True)

# --- get a rolling average of the last N days ---
# --- for temp/precip data by county ---

rollAvgTempWindow = 14
rollAvgPrecipWindow = 3
print("getting rolling averages. "
      f"Temperature roll window: {rollAvgTempWindow}"
      f"Precip roll window: {rollAvgPrecipWindow}")

# the data is sorted by county then date
# so the rolling average carries one county's data
# over to the next for the window of rolling (N).
# The first N days of a rolling avg should be NaN
# anyway so we allow this junk data and then replace
# with NaNs
rollAvgTempDF = fireWeatherDF['tavg'].rolling(rollAvgTempWindow).mean()
# find invalid dates by getting the first N dates
invalidRollDates = list(fireWeatherDF['date'].head(rollAvgTempWindow))
# create a mask that's true for every instance of those dates
rollMask = fireWeatherDF['date'].isin(invalidRollDates)
rollAvgTempDF = rollAvgTempDF.mask(rollMask)

fireWeatherDF['rtavg'] = rollAvgTempDF

rollAvgPrecipDF = fireWeatherDF['prcp'].rolling(rollAvgPrecipWindow).mean()
invalidRollDates = list(fireWeatherDF['date'].head(rollAvgPrecipWindow))
rollMask = fireWeatherDF['date'].isin(invalidRollDates)
rollAvgPrecipDF = rollAvgPrecipDF.mask(rollMask)
recentRain = [p > 0 if pd.notna(p) else np.NaN for p in rollAvgPrecipDF]

fireWeatherDF['rprcp'] = recentRain

# --- drop dates for which there are any temp NaNs ---

oldSize = len(fireWeatherDF)
fireWeatherDF.dropna(subset=['tavg', 'rtavg'], inplace=True)
print(f"dropped {oldSize - len(fireWeatherDF)} rows for NaN temps")

# --- bucket temp data ---
tempBucketSize = 5
precipBucketSize = 0.1
print("bucketing temperature and precip data."
      f"temp bucket size: {tempBucketSize} degrees,"
      f"precit bucket size: {precipBucketSize} inches"
      )

# bucket by subtracting temp % bucketSize, giving the nearest
# int divisible by bucketSize
fireWeatherDF['tavg'] = fireWeatherDF['tavg'].apply(lambda x: x - x % tempBucketSize)
fireWeatherDF['rtavg'] = fireWeatherDF['rtavg'].apply(lambda x: x - x % tempBucketSize)
fireWeatherDF['prcp'] = fireWeatherDF['prcp'].apply(lambda x: x - x % precipBucketSize)

fireWeatherDF.to_csv(OUTPUT_FILE, index=False)

# --- join fire info ---

print("joining fire data with weather date")

# start by converting string dates to datetime objects
# the fire data requires a format argument to parse, but
# the format is not a named argument so it cannot be passed
# directly to DF.apply()
fireStrToDT = lambda x: dt.datetime.strptime(x, '%m/%d/%y')

# add our columns for fireStarted and activeFires
fireWeatherDF['fireStarted'] = np.full(len(fireWeatherDF), fill_value=False)
fireWeatherDF['hasFire'] = np.full(len(fireWeatherDF), fill_value=False)
fireWeatherDF['activeFires'] = np.full(len(fireWeatherDF), fill_value=0)

# track what counties are/aren't in each dataset to ensure consistency
fireCounties = set()
allCounties = set(fireWeatherDF['county'])
for fdf in fireDFs:
    dtCols = fdf[['start','contained']].applymap(fireStrToDT)
    fdf[['start','contained']] = dtCols

    # lowercase everything for easier string comparison
    fdf['counties'] = fdf['counties'].apply(str.lower)
    # some fires occur in multiple counties, so the field
    # lists all counties separated by dashes
    fdf['counties'] = fdf['counties'].apply(str.split, sep='-')

    for row in fdf.itertuples():
        # each row in every fire dataframe represents a fire
        fireCounties.update(row.counties)
        # again, counties is a list
        for c in row.counties:
            # note that a fire started in this county on this date
            fireStartMask = (fireWeatherDF['county'] == c) \
                             & (fireWeatherDF['date'] == row.start)
            fireWeatherDF.loc[fireStartMask, 'fireStarted'] = True
            # get rows where county matches and dates are between
            # the fire starting and being contained
            activeFireMask = (fireWeatherDF['county'] == c) \
                              & (fireWeatherDF['date'] >= row.start) \
                              & (fireWeatherDF['date'] <= row.contained)
            newAF = fireWeatherDF.loc[activeFireMask, 'activeFires'].apply(lambda x: x + 1)
            fireWeatherDF.loc[activeFireMask, 'activeFires'] = newAF
            # we already have a mask for active fires, set hasFire to true there
            fireWeatherDF.loc[activeFireMask, 'hasFire'] = True

print("counties in fire data not in station data: ")
print(fireCounties - allCounties)
print("counties in station data with no fires: ")
print(allCounties - fireCounties)

# write out DF
print(fireWeatherDF.describe())

fireWeatherDF.to_csv(OUTPUT_FILE, index=False)
