import plotly

from fbprophet import Prophet
import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('fbprophet')
datas = collect_data_files('fbprophet')
# from when are we graphing
startdate = "20" + "15-1-1"
todaydate = "20" + str(date.today().strftime("%y-%m-%d"))

#####web app####
st.title = "yhfinance"

# what stocks are included
stocks = ("ACN","AAPL", "GOOG", "MSFT", "ABBV", "UPS", "WFC", "BAC","GME")

# elements
selected_stock = st.selectbox("Choose A Stock For Prediction", stocks)
s_years = st.slider("How Far Would You Like To Predict", 0, 4)
period = s_years * 365


####data download####
@st.cache
def dl_data(ticker):
    data = yf.download(ticker, startdate, todaydate)
    data.reset_index(inplace=True)
    return data


data_loading = st.text("Loading...")
data = dl_data(selected_stock)
data_loading.text("")

# display raw data#
st.subheader("RAW Stock Data")
st.write(data.tail())

# graph data
mode1 = ("Open", "Close", "High", "Low")
mode2 = ("Open", "Close", "High", "Low")
graph_mode1 = st.selectbox("select what you would like to graph", mode1)
graph_mode2 = st.selectbox("select what you would like compare it to", mode2)


def graph_raw():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[graph_mode1], name=selected_stock + " " + graph_mode1))
    fig.add_trace(go.Scatter(x=data['Date'], y=data[graph_mode2], name=selected_stock + " " + graph_mode2))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


graph_raw()

# predictions#
pmode = st.selectbox("Select what you would like to predict", mode1)
data_train = data[['Date', pmode]]
data_train = data_train.rename(columns={"Date": "ds", pmode: "y"})

dm = Prophet()
dm.fit(data_train)
future = dm.make_future_dataframe(periods=period)
predict = dm.predict(future)

# display forecast
st.subheader("RAW Stock Data")
st.write(predict.tail())

#plot forecast
st.write("prediction data")
fig1 = plot_plotly(dm,predict)
st.plotly_chart(fig1)

st.write("prediction components")
fig2 = dm.plot_components(predict)
st.write(fig2)