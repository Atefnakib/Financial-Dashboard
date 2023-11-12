# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD EXAMPLE - v3
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from io import BytesIO
import base64
from plotly.subplots import make_subplots




#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# Header
#==============================================================================
def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    
    # Create a Streamlit app
    st.title("Atef's Financial Dashboard 📈 ")

    col1, col2 = st.columns([1, 5])
    col1.write("Data source:")
    col2.image('./images/yahoo_finance.png', width=100)
    
    #here we defined the function to be able to download the file of the stocks as a csv
    def download_csv(data, filename):
        csv_file = data.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Get the list of stock tickers from S&P500
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

    # Add the ticker selection on the sidebar
    col1, col2 = st.columns(2)
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Ticker", ticker_list)

    # Add the "Update and Download Data" button
    col2.write("")  # Create some space
    if st.button("Update and Download Data"):
        if ticker:
            st.write(f"Updating stock data for {ticker}...")

            # Fetch stock data using yfinance
            stock_data = yf.download(ticker)

            # Display the updated data
            st.write(stock_data)

            # Download the data as a CSV file
            download_csv(stock_data, ticker)


#==============================================================================
# Tab 1: SUMMARY
#==============================================================================
def render_tab1(ticker):
    st.header("Summary")
    
    
    # Get the company information
    @st.cache_data
    def GetCompany(ticker):
        """
        This function gets the company information from Yahoo Finance.
        """
        return YFinance(ticker).info
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in dictionary format
        info = GetCompany(ticker)
        
        #here we are inserting the company summary
        st.write('**Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
            
        
        # Create a Ticker object for the stock
        stock_info = yf.Ticker(ticker)

    #here we are getting the information for the key statistics
        st.write('**Key Statistics**') 
        col1, col2 = st.columns([8,8])
        # Create dictionaries for stats in col1 and col2
        stats_col1 = {
            'Previous Close': info.get('previousClose'),
            'Open': info.get('open'),
            'Bid': info.get('bid'),
            'Ask': info.get('ask'),
            'Day Range': info.get('DAY_RANGE'),
            '52 Week Range': info.get('52weekrange'),
            'Volume': info.get('volume'),
            'Avg. Volume': info.get('avg.volume')
        }
        
        stats_col2 = {
            'Market Cap': info.get('marketCap'),
            'Beta (5Y Monthly)': info.get('beta (5Y Monthly)'),
            'PE Ratio (TTM)': info.get('peRatio (ttm)'),
            'EPS (TTM)': info.get('eps (ttm)'),
            'Earnings Date': info.get('earningsdate'),
            'Forward Dividend & Yield': info.get('forwarddividendandyield'),
            'EX-Dividend Date': pd.to_datetime(info.get('exDividendDate'), unit = 's'),
            '1Y Target Est': info.get('1ytargetest')
        }
        
        # Get the major shareholders information
        major_holders = stock_info.major_holders
            
        st.write('**Major Shareholders**')
        st.dataframe(major_holders)
            
        
        #here we are assigning what information we want in each column to have it as similar as Yahoo Finance
        with col1:
            # Show some statistics as a DataFrame
            company_stats_col1 = pd.DataFrame({'Value': pd.Series(stats_col1)})
            st.dataframe(company_stats_col1)
        
        with col2:
            # Show some statistics as a DataFrame
            company_stats_col2 = pd.DataFrame({'Value': pd.Series(stats_col2)})
            st.dataframe(company_stats_col2)
        
# here we need to insert the chart 
    # Plot the area chart 
    
    st.write('**Charts**')  
    
    stock_data =yf.download(tickers = ticker,period = "3y")
            
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1: date1 = st.button('1M')
    with col2: date2 = st.button('3M')
    with col3: date3 = st.button('6M')
    with col4: date4 = st.button('YTD')
    with col5: date5 = st.button('1Y')
    with col6: date6 = st.button('3Y')
    with col7: date7 = st.button('5Y')
    with col8: date8 = st.button('MAX')
            
    if date1:
            stock_data=yf.download(tickers = ticker,period = "1mo")
    if date2:
            stock_data=yf.download(tickers = ticker,period = "3mo")
    if date3:
            stock_data=yf.download(tickers = ticker,period = "6mo")
    if date4:
            stock_data=yf.download(tickers = ticker,period = "ytd")
    if date5:
            stock_data=yf.download(tickers = ticker,period = "1y")
    if date6:
            stock_data=yf.download(tickers = ticker,period = "3y")
    if date7:
             stock_data=yf.download(tickers = ticker,period = "5y")
    if date8:
            stock_data=yf.download(tickers = ticker)
            
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, 
                             y=stock_data['Close'], 
                             fill='tozeroy', 
                             name=ticker, 
                             line=dict(color='red')))
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
            )
            
    st.plotly_chart(fig, use_container_width=True)


#==============================================================================
# Tab 2: CHART
#==============================================================================
def render_tab2(ticker):
    st.header("Chart")

    col1, col2, col3 = st.columns([4, 4, 4])

    # Streamlit UI
    st.title('Stock Price Chart')

    with col1:
        selected_timeframe = st.selectbox('Select Timeframe:', ('1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y', 'MAX'))

    with col2:
        selected_interval = st.radio('Select Time Interval:', ('1d', '1mo', '1y'))

    with col3:
        selected_chart_type = st.radio('Select Chart Type:', ('Candle Plot', 'Line Plot'))

    end_date = pd.Timestamp.now()

    if selected_timeframe == '1M':
        start_date = end_date - pd.DateOffset(months=1)
    elif selected_timeframe == '3M':
        start_date = end_date - pd.DateOffset(months=3)
    elif selected_timeframe == '6M':
        start_date = end_date - pd.DateOffset(months=6)
    elif selected_timeframe == 'YTD':
        start_date = pd.Timestamp(datetime(end_date.year, 1, 1))
    elif selected_timeframe == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif selected_timeframe == 'MAX':  
        start_date = end_date - pd.DateOffset(years=50)
    else:
        start_date = end_date - pd.DateOffset(years=int(selected_timeframe[:-1]))

    stock_data = yf.download(ticker, start=start_date, end=end_date, interval=selected_interval)

    # Calculate moving average with a window of 50 days
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()

    max_price = stock_data[['Close', 'MA_50']].max().max()
    max_volume = stock_data['Volume'].max()
    scale_factor = max_price / max_volume
    scaled_volume = stock_data['Volume'] * scale_factor

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    if selected_chart_type == 'Candle Plot':
        fig.add_trace(go.Candlestick(x=stock_data.index,
                                     open=stock_data['Open'],
                                     high=stock_data['High'],
                                     low=stock_data['Low'],
                                     close=stock_data['Close'],
                                     name='Candlesticks'))
    else:
        fig.add_trace(go.Scatter(x=stock_data.index, 
                                 y=stock_data['Close'], 
                                 mode='lines', 
                                 name='Close Price'))
        
    # Add custom hover text for volume bars
    hover_text = [f'Volume: {volume}' for volume in stock_data['Volume']]
    
    fig.add_trace(go.Bar(x=stock_data.index, y=scaled_volume, name='Volume', hovertext=hover_text), secondary_y=False)

    # Plot the moving average
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MA_50'], mode='lines', name='MA (50 days)', line=dict(color='orange')))

    fig.update_yaxes(title_text='Price', secondary_y=False)
    fig.update_yaxes(title_text='Scaled Volume', secondary_y=True)

    fig.update_layout(
        title=f'{ticker} Stock Price Chart with 50-day Moving Average',
        xaxis_rangeslider_visible=True,
        xaxis_title='Date',
    )

    st.plotly_chart(fig)

#==============================================================================
# Tab 3: FINANCIALS
#==============================================================================
def render_tab3(ticker):
    st.header("Financials")

    #creating columns     
    col1, col2 = st.columns([4, 4])
    with col2: 
        #creating a bullet point choice for the annualy and quarterly financials 
        select_timeframe = st.radio("Selet the timeframe", ("Annualy", "Quarterly"))
    
    with col1:
        #creating a bullet point choice to pick the different financials 
        select_info = st.radio("Select Information", ("Income Statement", "Balance Sheet", "Cash Flow"))
    # here we created an if loop inside the if loop so that if annualy was chosen and Income statement, 
    #the end user will see the correct information which was taken from the y finance library
    #after each of these we put in a data frame to have them in a proper table        
    if select_timeframe == "Annualy":

        if select_info == "Income Statement":
            st.write('**Income Statement**')
            st.dataframe(yf.Ticker(ticker).income_stmt)
        elif select_info == "Balance Sheet":
          st.write('**Balance Sheet**')
          st.dataframe(yf.Ticker(ticker).balance_sheet)
        elif select_info == "Cash Flow":
          st.write('**Cash Flow**') 
          st.dataframe(yf.Ticker(ticker).cashflow)
    else:
        if select_info == "Income Statement":
            st.write('**Income Statement**')
            st.dataframe(yf.Ticker(ticker).quarterly_income_stmt)
        elif select_info == "Balance Sheet":
          st.write('**Balance Sheet**')
          st.dataframe(yf.Ticker(ticker).quarterly_balance_sheet)
        elif select_info == "Cash Flow":
          st.write('**Cash Flow**') 
          st.dataframe(yf.Ticker(ticker).quarterly_cashflow)


#==============================================================================
# Tab 4: Monte Carlo Simulation
#==============================================================================
def render_tab4(ticker):
    st.header("Monte Carlo Simulation")
    
    #created the colums 
    col1, col2 = st.columns([4, 4])
    
    #created a ticker to retrieve the stock info
    stock = yf.Ticker(ticker)
    
    #we got the historical price data
    historical_data = stock.history(period="5y") 
    
    
    #here we selected the time horizon and the number of simulation desired to be ran
    with col1:
        simulations = st.selectbox('Select the number of simulation:', (200, 500, 1000))
    with col2:
        time_horizon = st.selectbox('Select the time horizon:', (30, 60, 90))
        
        
        
    #Calculated daily returns
    daily_return = historical_data['Close'].pct_change()
    daily_volatility = daily_return.std()
    
    #got the closed price
    close_price = historical_data['Close']

    #created an empty data frame to input the data in 
    simulated_df = pd.DataFrame()
    
    #here we started with the monte carlo simulations
    #here from mthe variables from the drop down menus we created the for loops
        
    for r in range(simulations): #this is the number of simulations
        stock_price_list = []
        current_price = historical_data['Close'][-1]

        for i in range(time_horizon): #this is the time horizon
            daily_return = np.random.normal(0, daily_volatility, 1)[0]
            future_price = current_price * (1 + daily_return)
            stock_price_list.append(future_price)
            current_price = future_price 
    
    
        simulated_col = pd.Series(stock_price_list)
        simulated_col.name = "Sim" + str(r)
        simulated_df = pd.concat([simulated_df, simulated_col], axis=1)
    
    #with this code we plotted the results of the simulation
    st.pyplot(plot_simulation_price(simulated_df))
    
    #here you will find all the calculations and code to calculate the value at risk were it was rounded to make it easily readable
    ending_price = simulated_df.iloc[-1, :]
    future_price_95ci = np.percentile(ending_price, 5)
    VaR = close_price[-1] - future_price_95ci
    st.subheader(f'VaR at 95% confidence interval is: {np.round(VaR, 2)} USD')
    
    #here we created another function for the plotting where we input the information calculated previously in it 
def plot_simulation_price(simulated_df):
        fig, ax = plt.subplots()
        ax.plot(simulated_df)
        plt.title(f"Monte Carlo Simulation for {ticker} stock prices")
        plt.xlabel('Days')
        plt.ylabel('Price')
        return fig

#==============================================================================
# Tab 5: My Analysis
#==============================================================================
def render_tab5(ticker):

    #title and description
    st.title("Stock Investment Calculator")
    st.write("Enter your investment details to calculate profit, loss, and return on investment.")

    #inputing the initial investement amount
    initial_investment = st.number_input("Initial Investment Amount ($)", key="initial_investment", value=1000.0, min_value=0.01, step=0.10)

    #selecting the date of the investment
    investment_date = st.date_input("Date of Investment", key="investment_date", value=datetime.today() - timedelta(days=7)) 

    #including here thr global ticker to perform the calculations
    ticker_symbol = ticker.upper()

    #getting the stock information from YFinance library
    stock = yf.Ticker(ticker_symbol)
    historical_data = stock.history(start=investment_date, end=datetime.today())
    
    if not historical_data.empty:
        current_stock_price = historical_data.iloc[-1]['Close']
        #calculating the current value of the investment using the stock price on the day of investment
        current_investment_value = (initial_investment / historical_data.iloc[0]['Close']) * current_stock_price
    else:
        st.error("No data available for the selected date range. Please adjust the date.")
        return

    #calculating the profit or loss
    profit_or_loss = current_investment_value - initial_investment

    #calculating the return on investement
    roi = ((current_investment_value - initial_investment) / initial_investment) * 100

    # Display results
    st.write(f"**Stock Price on Investment Date:** ${historical_data.iloc[0]['Close']:.2f}")
    st.write(f"**Current Stock Price:** ${current_stock_price:.2f}")
    st.write(f"**Total Investment Cost:** ${initial_investment:.2f}")
    st.write(f"**Current Investment Value:** ${current_investment_value:.2f}")
    st.write(f"**Return on Investment (ROI):** {roi:.2f}%")

    if profit_or_loss > 0:
        st.success(f"**Profit:** ${profit_or_loss:.2f}")
    elif profit_or_loss < 0:
        st.error(f"**Loss:** ${-profit_or_loss:.2f}")
    else:
        st.write("No Profit, No Loss.")
#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

selected = st.selectbox("Select Section", ["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Stock Investement Calculator"])

# Render the tabs
if selected == "Summary":
    render_tab1(ticker)
elif selected == "Chart":
    render_tab2(ticker)
elif selected == "Financials":
    render_tab3(ticker)
elif selected == "Monte Carlo Simulation":
    render_tab4(ticker)
elif selected == "Stock Investement Calculator":
    render_tab5(ticker)

    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################