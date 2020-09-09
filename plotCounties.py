#! /usr/bin/python3
import sys
import plotly.express as px
import json
import pandas as pd
import datetime as dt

with open("CalCountyGeojson.json") as f:
    geoJson = json.load(f)

fireProbDF = pd.read_csv("bakedData/predictedFireData.csv", dtype={'fips':object})

# raw fire probability can oscillate a lot, a rolling average
# smooths it out and makes sense (probability should change pretty slowly)
rollAvgProbWindow = 7
rollAvgProbDF = fireProbDF['fireProb'].rolling(rollAvgProbWindow).mean()
# find invalid dates by getting the first N dates
invalidRollDates = list(fireProbDF['date'].head(rollAvgProbWindow))
# create a mask that's true for every instance of those dates
rollMask = fireProbDF['date'].isin(invalidRollDates)
validDateMask = ~rollMask
rollAvgProbDF = rollAvgProbDF.mask(rollMask)

fireProbDF['rFireProb'] = rollAvgProbDF

dates = set(fireProbDF.loc[validDateMask, 'date'])
i = 0
for date in sorted(dates):
    dateMask = fireProbDF['date'] == date
    probOnDate = fireProbDF.loc[dateMask]

    fig = px.choropleth_mapbox(probOnDate,
                               geojson=geoJson,
                               locations='fips',
                               color='fireProb',
                               color_continuous_scale='YlOrRd',
                               range_color=(0,1),
                               center={"lat": 37.5, "lon": -119.4179},
                               mapbox_style='white-bg',
                               zoom=4.9,
                               width=720,
                               height=720,
                               )

    # format the graph all pretty
    title = f"Predicted Probability of<br>Wildfire by County<br>{date}"
    fig.update_layout(title_text=title,
                      title_x=0.5,
                      font_family="Courier New",
                      title_font_family="Courier New",
                      title_font_size=24,
                      coloraxis_colorbar_title="wildfire<br>probability",
                      margin={'t': 50, 'b': 10, 'l': 0, 'r': 50},
                      )

    # output our progress
    i += 1
    sys.stdout.write(f"\r{i}/{len(dates)} dates done")
    sys.stdout.flush()

    # use epoch time for easier sorting
    dateEpoch = int(dt.datetime.fromisoformat(date).timestamp())
    fig.write_image("graphs/animFrames/animGraph_%04d.jpg" % i)