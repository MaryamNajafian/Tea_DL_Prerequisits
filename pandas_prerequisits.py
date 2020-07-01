"""
Pandas topics:
    Loading
    Selecting rows/cols
    Apply function
    Plotting
"""
#%% downloading data
import pandas as pd
import wget
url = 'https://github.com/lazyprogrammer/machine_learning_examples/blob/master/pytorch/sbux.csv'
wget.download(url)


#%% loading data
df = pd.read_csv('sbux.csv')
df = pd.read_csv(url)
#%% reading the data
df.head(10) # !head sbux.csv
df.tail(10)
df.info() #stats about the data

#%% selecting rows and cols

# show  all the column names
df.columns

# rename columns
df.columns = ['bla','blaaa','blaaa']

""" select the column/row
loc: we have to specify the `name` of the rows and columns that we need to filter out. it gets rows (or columns) with particular labels from the index. 
iloc: we have to specify rows/columns by their `integer index`. It gets rows (or columns) at particular positions in the index. iloc is integer index-based. 

"""
df.iloc[0] # used for integer indeces
df.loc[0] # used selects by index label
df.loc['2000-02-02'] # used selects by index label


#select a single column by name
df['bla']

#select multiple columns by name
df[['bla','blaa']]

# pandas series datatype (single column)
type(df['bla'])

# pandas dataframe datatype (multi columns)
type(df[['bla','blaa']])

#%%
# selecting rows based on their values on a specific column
df[df['bla']>700]
df[df['blaa'] != 'saturday']

#%%
# convert pandas to numpy array by using`.values`on columns with numerical values ONLY (not string)

A = df[['bla','blaa']].values
type(A)

#%% save portions of a vbig csv into a smaller csv file
smalldf = df[['bla','blaa']]
smalldf.to_csv('output.csv')

#%% apply() a function to rows/columns
# if you want to do the same opperation on each row/columns of the dataframe
# and helps to avoid the FOR LOOP

def data_to_year(row):
    return int(row['data'].split('-')[0]) #format 2000-02-02

df.apply(data_to_year, axis=1) #axis=1: operattes on every row

# save in a new column
df['blaaaa'] = df.apply(data_to_year, axis=1)

#%% plot dfs and series

# histogram
df['bla'].hist()

# line chart
df['bla'].plot()

# box plot [outlier...Min...25 percentile...Median...75 Percentile...Max...Outliers
df[['bla', 'blaa', 'blaaaa']].plot.box()

# scatter plot: it shows statistical summary of the data and it shows linear correlation between each of the data columns
from pandas.plotting import scatter_matrix
scatter_matrix(df[['bla', 'blaa', 'blaaaa']], alpha=0.2, figsize=(6,6))