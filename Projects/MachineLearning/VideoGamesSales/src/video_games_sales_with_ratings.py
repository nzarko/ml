# -*- coding: utf-8 -*-

# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We will use the Seaborn library
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg' 

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
plt.rcParams['image.cmap'] = 'viridis'


DATA_DIR = '/Data/'
import os
cur_dir = os.getcwd()
print(cur_dir)
os.listdir(cur_dir+DATA_DIR)
""" from pathlib import Path

mypath = Path().absolute()
print(mypath) """



# Import the dataset - ignore NaN's
dataset = pd.read_csv(cur_dir+DATA_DIR+'Video_Games_Sales_as_at_22_Dec_2016.csv').dropna()

dataset.shape
# Get info about data
dataset.info()

# We see that pandas has loaded some of the numerical features as object type. 
# We will explicitly convert those columns into float and int.

dataset['User_Score'] = dataset['User_Score'].astype('float64')
dataset['Year_of_Release'] = dataset['Year_of_Release'].astype('int64')
dataset['User_Count'] = dataset['User_Count'].astype('int64')
dataset['Critic_Count'] = dataset['Critic_Count'].astype('int64')

dataset.info()
useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 
               'Global_Sales', 'Critic_Score', 'Critic_Count',
               'User_Score', 'User_Count', 'Rating'
              ]
dataset[useful_cols].head()

dataset[[x for x in dataset.columns if 'Sales' in x] + 
   ['Year_of_Release']].groupby('Year_of_Release').sum().plot();

dataset[[x for x in dataset.columns if 'Sales' in x] + 
   ['Year_of_Release']].groupby('Year_of_Release').sum().plot(kind='bar',rot=45);

pairplot()

## Let's take a look at the first of such complex plots, 
# a pairwise relationships plot, which creates a matrix 
# of scatter plots by default. This kind of plot helps 
# us visualize the relationship between different variables 
# in a single output.
# `pairplot()` may become very slow with the SVG format
%config InlineBackend.figure_format = 'png' 
sns.pairplot(dataset[['Global_Sales', 'Critic_Score', 'Critic_Count', 
                 'User_Score', 'User_Count']]);

sns.pairplot(dataset[['Global_Sales','Critic_Score']])

%config InlineBackend.figure_format = 'svg'
sns.distplot(dataset['Critic_Score']);

sns.jointplot(x='Critic_Score', y='User_Score', 
              data=dataset, kind='scatter');

# boxplot
top_platforms = dataset['Platform'].value_counts().sort_values(ascending=False).head(5).index.values
sns.boxplot(y="Platform", x="Critic_Score", 
            data=dataset[dataset['Platform'].isin(top_platforms)], orient="h");

# It is worth spending a bit more time to discuss how to
# interpret a box plot. Its components are a box 
# (obviously, this is why it is called a box plot), the 
# so-called whiskers, and a number of individual points 
# (outliers).
# The box by itself illustrates the interquartile spread 
# of the distribution; its length determined by the 
# 25%(Q1) and 75%(Q3) percentiles. The vertical line 
# inside the box marks the median (50%) of the 
# distribution. 
# The whiskers are the lines extending from the box. 
# They represent the entire scatter of data points, 
# specifically the points that fall within the interval 
# (Q1−1.5⋅IQR,Q3+1.5⋅IQR), where IQR=Q3−Q1 is the 
# interquartile range.
# Outliers that fall out of the range bounded by the 
# whiskers are plotted individually.


# heatmap()
# The last type of plot that we will cover here is a 
# heat map. A heat map allows you to view the 
# distribution of a numerical variable over two 
# categorical ones. Let’s visualize the total sales of 
# games by genre and gaming platform.
platform_genre_sales = dataset.pivot_table(
                        index='Platform', 
                        columns='Genre', 
                        values='Global_Sales', 
                        aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5);

# Plotly
# We have examined some visualization tools based on the
# matplotlib library. However, this is not the only 
# option for plotting in Python. Let’s take a look at the
# plotly library. Plotly is an open-source library that 
# allows creation of interactive plots within a Jupyter 
# notebook without having to use Javascript.
# The real beauty of interactive plots is that they 
# provide a user interface for detailed data exploration. 
# For example, you can see exact numerical values by 
# mousing over points, hide uninteresting series from 
# the visualization, zoom in onto a specific part of the 
# plot, etc.
# Before we start, let’s import all the necessary modules 
# and initialize plotly by calling the init_notebook_mode() 
# function.

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)


# Line plot
# First of all, let’s build a line plot showing the 
# number of games released and their sales by year.
years_df = dataset.groupby('Year_of_Release')[['Global_Sales']].sum().join(
    dataset.groupby('Year_of_Release')[['Name']].count())
years_df.columns = ['Global_Sales', 'Number_of_Games']



# Figure is the main class and a work horse of 
# visualization in plotly. It consists of the data 
# (an array of lines called traces in this library) and 
# the style (represented by the layout object). In the 
# simplest case, you may call the iplot function to 
# return only traces.
# The show_link parameter toggles the visibility of the 
# links leading to the online platform plot.ly in your 
# charts. Most of the time, this functionality is not 
# needed, so you may want to turn it off by passing 
# show_link=False to prevent accidental clicks on 
# those links.
# Create a line (trace) for the global sales
trace0 = go.Scatter(
    x=years_df.index,
    y=years_df['Global_Sales'],
    name='Global Sales'
)

# Create a line (trace) for the number of games released
trace1 = go.Scatter(
    x=years_df.index,
    y=years_df['Number_of_Games'],
    name='Number of games released'
)

# Define the data array
data = [trace0, trace1]

# Set the title
layout = {'title': 'Statistics for video games'}

# Create a Figure and plot it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)

# As an option, you can save the plot in an html file:
plotly.offline.plot(fig, filename='years_stats.html', show_link=False);


# Bar chart
# Let's use a bar chart to compare the market share of 
# different gaming platforms broken down by the number 
# of new releases and by total revenue.
# Do calculations and prepare the dataset
platforms_df = dataset.groupby('Platform')[['Global_Sales']].sum().join(
    dataset.groupby('Platform')[['Name']].count()
)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)

# Create a bar for the global sales
trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df['Global_Sales'],
    name='Global Sales'
)

# Create a bar for the number of games released
trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df['Number_of_Games'],
    name='Number of games released'
)

# Get together the data and style objects
data = [trace0, trace1]
layout = {'title': 'Market share by gaming platform'}

# Create a `Figure` and plot it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)


# Box plot
# plotly also supports box plots. Let’s consider the 
# distribution of critics' ratings by the genre of the 
# game.
data = []

# Create a box trace for each genre in our dataset
for genre in dataset.Genre.unique():
    data.append(
        go.Box(y=dataset[dataset.Genre == genre].Critic_Score, name=genre)
    )
    
# Visualize
iplot(data, show_link=False)