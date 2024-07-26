import datetime
import streamlit as st
#import pandas as pd
#import cufflinks as cf
import yfinance as yf
import pandas-datareader as pdr
yf.pdr_override()
import numpy as np

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

APP_NAME = "Stock App!"
# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write(' ### Stocks Prediction App ')

# Add app title
st.sidebar.title(APP_NAME)

TICKERS = ['SHB-A.ST','SEB-A.ST','SWED-A.ST','ASAZY','HM-B.ST','VLVLY','SPOT','ERIC']
ticker = st.sidebar.selectbox('Select ticker', sorted(TICKERS), index=0)

# Set start and end point to fetch data
start_date = st.sidebar.date_input('Start date', datetime.datetime(2008, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

#df = yf.download(ticker, start_date, end_date,threads=False)
#df = yf.download(ticker, start_date, end_date,threads=False)
df = pdr.get_data_yahoo(ticker, start_date, end_date, threads=False)
df.reset_index(inplace=True)
st.subheader(f'{ticker} Stock Price')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

df['sma'] = df['Close'].rolling(20).mean()  # 20 day moving average
df['lma']=  df['Close'].rolling(200).mean()  # 200 day moving average
# Plot the data along with the short and long moving averages of 20 and 200 days 

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['sma'],line_color='red' ,name="20 day moving average"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['lma'], name="200 day moving average"))
fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True,autosize=False,width=800,height=600)
st.plotly_chart(fig)

st.subheader('Descriptive Statistics')
st.write(df.describe())


n_years = st.selectbox('Years of prediction:', (1,2,3))
period = n_years * 365

# Predict forecast with Prophet.
df_train = df[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# plot the forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)


st.subheader('**Bollinger Bands**')
st.markdown('Bollinger bands help determine whether prices are high or low on a relative basis. When the bands tighten during a period of low volatility, it raises the likelihood of a sharp price move in either direction. This may begin a trending move.When the bands separate by an unusual large amount, volatility increases and any existing trend may be ending. ')
WINDOW = 20
df['sma'] = df['Close'].rolling(WINDOW).mean()
df['std'] = df['Close'].rolling(WINDOW).std(ddof = 0)

import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots

# Create subplots with 2 rows; top for candlestick price, and bottom for bar volume
fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, subplot_titles = (ticker, 'Volume'), vertical_spacing = 0.1, row_width = [0.2, 0.7])

# ----------------
# Candlestick Plot
fig.add_trace(go.Candlestick(x = df['Date'],
                             open = df['Open'],
                             high = df['High'],
                             low = df['Low'],
                             close = df['Close'], showlegend=False,
                             name = 'candlestick'),
              row = 1, col = 1)

# Moving Average
fig.add_trace(go.Scatter(x = df['Date'],
                         y = df['sma'],
                         line_color = 'black',
                         name = 'sma'),
              row = 1, col = 1)

# Upper Bound
fig.add_trace(go.Scatter(x = df['Date'],
                         y = df['sma'] + (df['std'] * 2),
                         line_color = 'magenta',
                         line = {'dash': 'dash'},
                         name = 'upper band',
                         opacity = 0.5),
              row = 1, col = 1)

# Lower Bound fill in between with parameter 'fill': 'tonexty'
fig.add_trace(go.Scatter(x = df['Date'],
                         y = df['sma'] - (df['std'] * 2),
                         line_color = 'grey',
                         line = {'dash': 'dash'},
                         fill = 'tonexty',
                         name = 'lower band',
                         opacity = 0.5),
              row = 1, col = 1)


# ----------------
# Volume Plot
fig.add_trace(go.Bar(x = df['Date'], y = df['Volume'], showlegend=False), 
              row = 2, col = 1)

# Remove range slider; (short time frame)
fig.update(layout_xaxis_rangeslider_visible=False)

st.plotly_chart(fig)

