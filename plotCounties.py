#! /usr/bin/python3

import plotly.express as px
import json
import pandas as pd
from random import randint

with open("CalCountyGeojson.json") as f:
    geoJson = json.load(f)

stations = pd.read_csv("cleanedData/geocodedStations.csv", dtype={'fips':object})
df = pd.DataFrame(stations['fips'])

df['val'] = [randint(0, 20) for _ in range(len(df))]
colorscale = ["#%06x" % randint(0, 0xffffff) for _ in range(len(df))]

fig = px.choropleth_mapbox(df,
                    geojson=geoJson,
                    locations='fips',
                    color='val',
                    color_continuous_scale='YlOrRd',
                    range_color=(0,20),
                    center={"lat": 37.5, "lon": -119.4179},
                    mapbox_style='white-bg',
                    zoom=5,
                    width=1280,
                    height=720
)
#fig.show()
fig.write_image("foo.jpg")
