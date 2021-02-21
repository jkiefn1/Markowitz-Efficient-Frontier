## ~~~~~~~~~~~~~~~~ ##
## ~~ DISCLAIMER ~~ ##
## ~~~~~~~~~~~~~~~~ ##

# THIS PYTHON PROGRAM WAS CREATED BY AN AMATEUR WITH NO PROFESSIONAL FINANCE BACKGROUND

# DO NOT RELY ON THIS PROGRAM'S ACCURACY OR CORRECTNESS IN ANY MANNER

# OWNER DOES NOT GUARANTEE ANY ACCURACY OR CORRECTNESS

# THIS PROGRAM IS FOR ILLUSTRATIVE PURPOSES ONLY

##### ~~~~~~~~~~~~~~~~#####
## ~~ ACKNOWLEDGEMENT ~~ ##
##### ~~~~~~~~~~~~~~~ #####

# Do you acknowledge that the script is done by an amateur,
# ...and its accuracy is not guaranteed?

# Change to 'yes' to ackowledge and be able to run the program
Acknowledgement = 'No'


##### ~~~~~~~~~~~~~~~~#####
## ~~~~ CODE BEGINS ~~~~ ##
##### ~~~~~~~~~~~~~~~ #####

if Acknowledgement.lower() == 'no':
    print('You did not ackwowledge this program was done by a non-finance professional and an amatuer, and therefore you will not be allowed to run the program.')
elif Acknowledgement.lower() == 'yes':

    import pandas as pd
    import numpy as np
    import datetime as dt
    from pandas_datareader import data as pdr
    import yfinance as yf
    import random
    from pandas.plotting import register_matplotlib_converters
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go

    register_matplotlib_converters()
    yf.pdr_override()

    # Create a timeframe for review
    amnt_years = 5
    start = dt.datetime.today() - dt.timedelta(days=amnt_years*365)
    end = dt.datetime.today()

    # List your Mutual Fund, Index Fund, Stock, etc. tickers
    funds =   [
            # list
            # your
            # stocks or mutual funds
            # here
            ]
      
    # Set a benchmark    
    index = [
        # enter *one* benchmark fund here
        ]

    # Desired number of holdings
    amnt_holdings = 3

    # Set up the dataframe
    prices = []
    returns = []

    for ticker in funds:
        print('Fetching {}'.format(ticker))

        # Get the prices and the daily returns of that fund
        price = pdr.get_data_yahoo(ticker, start, end)[['Adj Close']].rename(columns = {'Adj Close': ticker})
        ret = np.log(price/price.shift(1))
        
        if ticker == funds[0]:
            df_returns = pd.DataFrame(ret)
        else:
            df_temp2 = pd.DataFrame(ret)
            df_returns = df_returns.join(df_temp2, how='outer', lsuffix='_left', rsuffix='_right')
            
    # Do the same for the benchmark for later use
    for ticker in index:
        print('Fetching the benchmark, {}'.format(ticker))

        # Get the prices and the daily returns of that fund
        price = pdr.get_data_yahoo(ticker, start, end)[['Adj Close']].rename(columns = {'Adj Close': ticker})
        ret = np.log(price/price.shift(1))
        
        if ticker == index[0]:
            df_benchmark_returns = pd.DataFrame(ret)
        else:
            df_temp2 = pd.DataFrame(ret)
            df_benchmark_returns = df_benchmark_returns.join(df_temp2, how='outer', lsuffix='_left', rsuffix='_right')
            

    # ~~~~ INTERACTIVE FUND CORRELATION HEATMAP ~~~~

    # Create the correlation matrix
    corr = df_returns.corr()

    # Create the figure
    fig_hm = px.imshow(corr, 
                    labels=dict(x='Fund A', y='Fund B', color="Correlation"))

    fig_hm.update_layout(title="Correlation bewteen Selected Funds",title_x=0.5)

    fig_hm.show()

    # ~~~~~~~ INTERACTIVE FUND RETURNS ~~~~~~~
    # ~~~~ HYPOTHETICAL GROWTH OF $1,000 ~~~~

    # Create a growth DataFrame
    df_growth = round(((df_returns + 1).cumprod() * 10000),2)

    # Create the figure
    fig_1 = go.Figure()

    # add a scatter trace for every column
    for col in df_returns.columns:
        fig_1.add_scatter(x=df_growth.index, y=df_growth[col], name=col)

    # change the scale to logarithmic and add title
    fig_1.update_layout(title="Growth of $1,000 from {} - ".format(df_growth.index[0].strftime('%b %Y')) + "{}".format(df_growth.index[-1].strftime('%b %Y')), title_x=0.5)

    fig_1.update_layout(hovermode="x unified", yaxis_tickformat = '$,')

    fig_1.show()

    # ~~~~ CREATE THE SIMULATION NUMPY ARRAY ~~~~

    # Initialize things
    np.random.seed(1)
    N = df_returns.shape[0] # length of the returns DataFrame
    L = len(funds)
    RFR = 0 # Risk-free returns, for Sharpe Ratio Calc
    num_runs = 1000
    num_runs = num_runs+1

    # Set up all of the arrays
    port_funds = np.zeros((num_runs, L)).astype(str)
    port_weights = port_weights = np.zeros((num_runs, L))  
    np_final_info = np.zeros((num_runs, L+3))
    np_final_ret = np.zeros((N, num_runs))
    ret_arr = np.zeros(num_runs)
    vol_arr = np.zeros(num_runs)
    sharpe_arr = np.zeros(num_runs)

    print('Starting the portfolio optimiziation.....')

    for x in range(num_runs):
        
        if x%1000 == 0 and x > 0:
            print('{} portfolios simulated'.format(x))
        else:
            pass
        
        # Daily returns of random N holdings from original list
        port_funds[x,:] = df_returns.columns.tolist()
        mean = df_returns.mean()*252
        covariance = df_returns.cov()*252
        
        # Randomize weightings for funds; nums for amnt_holdings, 0s for rest
        weights_init = np.array([0] * (L - amnt_holdings) + [random.random() for i in range (amnt_holdings)])
        random.shuffle(weights_init)
        weights = np.round(weights_init/np.sum(weights_init), decimals = 2)
        port_weights[x,:] = weights
        
        # Expected final return
        ret_arr[x] = round(np.sum((mean * weights)), 3)
        
        # Expected return profile (by day)
        df_port = ((df_returns + 1).cumprod() * 10000)
        df_port = (df_port * weights)                                       
        np_final_ret[:,x] = round(np.sum(df_port, axis=1),2)
        
        # Expected volatility
        vol_arr[x] = round(np.sqrt(np.dot(weights.T, np.dot(covariance, weights))),3)
        
        # Sharpe Ratio
        sharpe_arr[x] = round(((ret_arr[x]-RFR)/vol_arr[x]),3)

        # Final Array
        np_final_info[x,:] = np.concatenate((port_weights[x], ret_arr[x], vol_arr[x], sharpe_arr[x]), axis=None)
        
    print('Done simulating. {} portfolios simulated'.format(x))
        
    # Transform the arrays into a DataFrame
    df_info = pd.DataFrame(np_final_info, columns=(df_returns.columns.tolist()+['Returns','Volatility','Sharpe Ratio'])) 
    df_info = df_info.sort_values(by='Sharpe Ratio', ascending=False)

    df_return_profiles = pd.DataFrame(np_final_ret, index=df_returns.index)

    # Add a columns to the information DataFrame outlining what funds are held
    r,c = df_info.shape

    holdings = []

    for x in range(0,r):
        
        h = []
        
        for y in range(0,L):
            
            if df_info.iloc[x][y] == 0:
                pass
            else:
                h.append(' ' + str(round(df_info.iloc[x][y]*100)) + '% ' + str(df_info.columns[y]))
            
        holdings.append(', '.join(h))

    df_info['Holdings'] = holdings

    # Create the efficient frontier figure
    fig_ef = px.scatter(df_info
                     ,x='Volatility'
                     ,y='Returns'
                     ,color='Sharpe Ratio'
                     ,hover_data=['Holdings']
                    )

    fig_ef.update_layout(title='Efficient Frontier of {} Simulated Portfolios'.format(len(ret_arr)-1), title_x=0.5)

    fig_ef.show()

    # ~~~~~~~~~~~~~~~~~ INTERACTIVE FUND RETURNS ~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ HYPOTHETICAL GROWTH OF $1,000 FOR TOP SIMULATED PORTFOLIOS ~~~~

    # Create the figure
    fig_2 = go.Figure()

    # Get the top 5 portfolios by Sharpe Ratio
    top_sr = df_info[0:5].index.values

    # Add a scatter line for those portfolios
    for col in top_sr:
        fig_2.add_scatter(x=df_return_profiles.index, y=df_return_profiles[col], name=str(df_info.iloc[int(col)]['Holdings']))
        
    # Get the top 5 portfolios by Returns
    top_ret = df_info.sort_values('Returns', ascending=False)[0:5].index.values

    # Add a scatter line for those portfolios
    for col in top_ret:
        fig_2.add_scatter(x=df_return_profiles.index, y=df_return_profiles[col], name=str(df_info.iloc[int(col)]['Holdings']))
        
    # Calculate the growth of the benchmark, then add to plot for comparison
    df_bench = round(((df_benchmark_returns + 1).cumprod() * 10000),2)
    for col in df_bench:
        fig_2.add_scatter(x=df_bench.index, y=df_bench[col], name='Benchmark ({})'.format(str(col)), marker_color='black')

    # change the scale to logarithmic and add title
    fig_2.update_layout(title="Growth of $10,000 from {} - ".format(df_return_profiles.index[0].strftime('%b %Y')) + "{}".format(df_return_profiles.index[-1].strftime('%b %Y')), title_x=0.5)

    fig_2.update_layout(hovermode="x unified", yaxis_tickformat = '$,')

    fig_2.update_layout(legend=dict(font=dict(size=8,color="black"),))

    fig_2.show()
