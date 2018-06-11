# May first need:
# In your VM: sudo apt-get install libgeos-dev (brew install on Mac)
# pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import os

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

"""
IMPORTANT
This is EXAMPLE code.
There are a few things missing:
1) You may need to play with the colors in the US map.
2) This code assumes you are running in Jupyter Notebook or on your own system.
   If you are using the VM, you will instead need to play with writing the images
   to PNG files with decent margins and sizes.
3) The US map only has code for the Positive case. I leave the negative case to you.
4) Alaska and Hawaii got dropped off the map, but it's late, and I want you to have this
   code. So, if you can fix Hawaii and Alask, ExTrA CrEdIt. The source contains info
   about adding them back.
"""



"""
PLOT 1: SENTIMENT OVER TIME (TIME SERIES PLOT)
"""
# Assumes a file called time_data.csv that has columns
# date, Positive, Negative. Use absolute path.

ts = pd.DataFrame()
directory = "/home/cs143/data/time_data.csv"
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            file = directory + "/" + file
            ts = pd.read_csv(file)
            break

ts.columns = ['date', 'Positive', 'Negative']
# Remove erroneous row.
ts = ts[ts['date'] != '2018-12-31']

plt.figure(figsize=(12,5))
ts.date = pd.to_datetime(ts['date'], format='%Y-%m-%d')
ts.set_index(['date'],inplace=True)

ax = ts.plot(title="President Trump Sentiment on /r/politics Over Time",
        color=['green', 'red'],
       ylim=(0, 1.05))
ax.plot()
plt.savefig("part1.png")
plt.close()

"""
PLOT 2: SENTIMENT BY STATE (POSITIVE AND NEGATIVE SEPARATELY)
# This example only shows positive, I will leave negative to you.
"""

# This assumes you have a CSV file called "state_data.csv" with the columns:
# state, Positive, Negative
#
# You should use the FULL PATH to the file, just in case.

directory = "/home/cs143/data/state_data.csv"
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            file = directory + "/" + file
            state_data = pd.read_csv(file)
            break

state_data.columns = ['state', 'Positive', 'Negative']

"""
You also need to download the following files. Put them somewhere convenient:
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf
https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx
IF YOU USE WGET (CONVERT TO CURL IF YOU USE THAT) TO DOWNLOAD THE ABOVE FILES, YOU NEED TO USE 
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shp?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.dbf?raw=true"
wget "https://github.com/matplotlib/basemap/blob/master/examples/st99_d00.shx?raw=true"
The rename the files to get rid of the ?raw=true
"""

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49,
        projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
shp_info = m.readshapefile('st99_d00','states',drawbounds=True)  # No extension specified in path here.
pos_data = dict(zip(state_data.state, state_data.Positive))
neg_data = dict(zip(state_data.state, state_data.Negative))

# choose a color for each state based on sentiment.
pos_colors = {}
statenames = []
pos_cmap = plt.cm.Greens 

vmin = state_data['Positive'].min(); vmax = state_data['Positive'].max() # set range.
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        pos = pos_data[statename.lower()]
        pos_colors[statename] = pos_cmap(( pos - vmin )/( vmax - vmin))[:3]
    statenames.append(statename)
# cycle through state names, color each one.

def alaska (x):
    return (0.35*x[0] + 1000000, 0.35*x[1]-1500000)

def hawaii (x):
    return (x[0] + 5100000, x[1]-1400000)

# POSITIVE MAP
ax = plt.gca() # get current axes instance
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
        # Alaska is too big. Scale it down to 35% first, then transate it. 
            seg = list(map(alaska, seg))
        if statenames[nshape] == 'Hawaii':
            seg = list(map(hawaii, seg))
        color = rgb2hex(pos_colors[statenames[nshape]]) 
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)
        
plt.title('Positive Trump Sentiment (darker is more positive)')
plt.savefig("posmap.png")

# NEGATIVE MAP
neg_colors = {}
statenames = []
neg_cmap = plt.cm.hot

vmin = state_data['Negative'].min(); vmax = state_data['Negative'].max() # set range.

for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia', 'Puerto Rico']:
        neg = neg_data[statename.lower()]
        neg_colors[statename] = neg_cmap(1. - ( neg - vmin )/( vmax - vmin))[:3]
    statenames.append(statename)
# cycle through state names, color each one.                                                                                                            

ax = plt.gca() # get current axes instance 
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
	# Alaska is too big. Scale it down to 35% first, then transate it.
            seg = list(map(alaska, seg))
        if statenames[nshape] == 'Hawaii':
            seg = list(map(hawaii, seg))
        color = rgb2hex(neg_colors[statenames[nshape]])
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)

plt.title('Negative Trump Sentiment (darker is more negative)')
plt.savefig("negmap.png")

# SOURCE: https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
# (this misses Alaska and Hawaii. If you can get them to work, EXTRA CREDIT)

# DIFFERENCE MAP

#we do Negative-Positive since the second term is always smaller
state_data['Difference'] = state_data['Negative'] - state_data['Positive']
diff_data = dict(zip(state_data.state, state_data.Difference))

vmin = state_data['Difference'].min()
vmax = state_data['Difference'].max()

diff_colors = {}
statenames = []
diff_cmap = plt.cm.hot

for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.                                                              
    if statename not in ['District of Columbia', 'Puerto Rico']:
        diff = diff_data[statename.lower()]
        diff_colors[statename] = diff_cmap(1. - ( diff - vmin )/( vmax - vmin))[:3]
    statenames.append(statename)
    # cycle through state names, color each one.
        
ax = plt.gca() # get current axes instance                                          
for nshape, seg in enumerate(m.states):
    # skip Puerto Rico and DC                                                                                            
    if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
        if statenames[nshape] == 'Alaska':
	# Alaska is too big. Scale it down to 35% first, then transate it.    
            seg = list(map(alaska, seg))
        if statenames[nshape] == 'Hawaii':
            seg = list(map(hawaii, seg))
        color = rgb2hex(diff_colors[statenames[nshape]])
        poly = Polygon(seg, facecolor=color, edgecolor=color)
        ax.add_patch(poly)

plt.title('Trump Sentiment Difference (darker is more negative)')
plt.savefig("diffmap.png")
plt.close()

"""
PART 4 SHOULD BE DONE IN SPARK
"""

"""
PLOT 5A: SENTIMENT BY STORY SCORE
"""
# What is the purpose of this? It helps us determine if the story score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called submission_score.csv with the following coluns
# submission_score, Positive, Negative

directory = "/home/cs143/data/story_score.csv"
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            file = directory + "/" + file
            story = pd.read_csv(file)
            break

story.columns = ['submission_score', 'Positive', 'Negative']

plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['submission_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['submission_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Submission Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5a.png")
plt.close()

"""
PLOT 5B: SENTIMENT BY COMMENT SCORE
"""
# What is the purpose of this? It helps us determine if the comment score
# should be a feature in the model. Remember that /r/politics is pretty
# biased.

# Assumes a CSV file called comment_score.csv with the following columns
# comment_score, Positive, Negative

directory = "/home/cs143/data/comment_score.csv"
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".csv"):
            file = directory + "/" + file
            story = pd.read_csv(file)
            break

story.columns = ['comment_score', 'Positive', 'Negative']

plt.figure(figsize=(12,5))
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(story['comment_score'], story['Positive'], s=10, c='b', marker="s", label='Positive')
ax1.scatter(story['comment_score'], story['Negative'], s=10, c='r', marker="o", label='Negative')
plt.legend(loc='lower right');

plt.xlabel('President Trump Sentiment by Comment Score')
plt.ylabel("Percent Sentiment")
plt.savefig("plot5b.png")
