import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import cvxopt
from scipy.stats import norm
import scipy.optimize
import scipy.sparse as sc
import scipy.stats
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

'''
Contains functions used in the Mathematical Finance.
'''

def read_return_history(ticker, start, end, interval = '1wk'):
    """ 
    Getting Returns data from Yahoo finance.
                    
            Parameters:
                    ticker (string) : List of Stock tickers.
                    start (datetime) :  Start date of stock data to be extracted.
                    end (datetime) :  End date of stock data to be extracted.
                    interval (string) : Interval of stock data to be extracted (default is '1wk').

            Returns:
                    return_history (double) : 2D Numpy array of stock returns data.
                    meanReturns (double) : 1D Numpy array of mean returns for each stock.
                    covMatrix (double): 2D Numpy array of Covariance Matrix(SIGMA).
    """
    # Create an empty DataFrame to store the Weekly Stock Price Data.
    stock_data = pd.DataFrame()

    # Fetch and append weekly data for each stock to the DataFrame.
    for tkr in ticker:
        stock = yf.download(tkr, start= start, end= end, interval= interval, progress=False);
        stock = stock[['Adj Close']]  # Selecting only Adj Close column
        stock.columns = [tkr]  # Rename column to stock symbol
        stock_data = pd.concat([stock_data, stock], axis=1)

    # Reset index to have a clean sequential index
    stock_data.reset_index(inplace=True)
    # Dealing with missing values.
    stock_data = stock_data.ffill()
    stock_data = stock_data.bfill()
    
    # Convert stock_data to a NumPy array
    price_history = stock_data.iloc[:, 1:].to_numpy()

    # Reverse the order of rows in the NumPy array to have the correct chronological order.
    price_history = price_history[::-1]
    
    # Computing Historical weekly returns.
    start_prices = price_history[:-1,:]    # All prices except last row.
    end_prices = price_history[1:,:]     # All prices except first row.
    return_history = (end_prices-start_prices)/start_prices
    meanReturns = np.mean(return_history, axis=0)
    covMatrix = np.cov( return_history, rowvar=False )
    return return_history, meanReturns, covMatrix

def portfolioMetric(weights, meanReturns, covMatrix, Time, riskfreerate = 0.05):
    ''' 
    Returns Expected Portfolios returns and Standard deviation over a specified time period. Also returns Sharpe Ratio of Portfolio.
    
            Parameters:
                    weights (string) : 1D Numpy array of weights of portfolio.
                    meanReturns (double) : 1D Numpy array of mean returns for each stock.
                    covMatrix (double): 2D Numpy array of Covariance Matrix(SIGMA).
                    Time (int) : Time given as number of days.
                    riskfreerate (double) : Risk Free Rate(default is 5%).

            Returns:
                    returns (double) : Expected return of the portfolio.
                    std (double) : Standard deviation of portfolio.
                    sharpe_ratio (double) : Sharpe Ratio of Portfolio.
    '''
    returns = (np.transpose(meanReturns) @ weights) * Time
    std = np.sqrt(np.transpose(weights) @ covMatrix @ weights) * np.sqrt(Time)
    sharpe_ratio = (returns - riskfreerate) / std
    return returns, std, sharpe_ratio

def get_call_option_data(ticker, date = '2024-09-20'):
    """ 
    Retrieving options data from Yahoo finance.
                    
            Parameters:
                    ticker (string) : Option ticker.
                    date (string) :  Expiration date of Option (default is '2024-09-20').
                    
            Returns:
                    option (pandas dataframe) : Pandas dataframe containing Option information.
    """
    tkr = yf.Ticker(ticker)
    option = tkr.option_chain(date)  # Option chain for a specific expiration date.
    option = option.calls 
    option['midPrice'] = (option['bid'] + option['ask']) / 2  # Average of Bid and Ask price.
    return option   

def markowitz_solver( sigma, mu, R ):
    '''
    Solves Markowitz Optimisation problem and returns a dictionary containing optimal weights in
    Markowitz Problem and the minimum Standard Deviation(Risk) that is achieved.

            Parameters:
                    sigma (double) : Return Covariance Matrix.
                    mu (double) :  Mean Returns.
                    R (double) : Target Expected Rate of Returns.

            Returns:
                    ret (double): Dictionary containing two keys weights and Standard deviation (sd).
    '''
    n = len(mu)
    P = 2*sigma
    q = np.zeros(n)  # zero arrays.
    A = np.array([np.ones(n), mu])
    b = np.array([1,R]).T
    res = cvxopt.solvers.qp(
        cvxopt.matrix(P),
        cvxopt.matrix(q), # q
        None, # G
        None, # h
        cvxopt.matrix(A),
        cvxopt.matrix(b))  
     
    # Res is a map with keys, status, x and primal objective.
    status = res['status']
    assert status=="optimal"  # Checking if Optimisation succeeded.
    
    w = res['x']  # Optimal value of x/w in Quadratic/Markowitz problem.
    var = res['primal objective']  # Minimum Variance acheived.
    
    sd = np.sqrt(var)  # Minimum Standard Deviation(risk).
    # Creating Dictionary of weights and standard deviation.
    ret = {}
    ret['weights']=w
    ret['sd']=sd
    
    return ret

def simulate_gbm( S0, mu, sigma, T, n_steps):
    '''
    Simulation of Geometric Brownian Motion(GBM) model to generate Stock Prices in discrete time.

            Parameters:
                    S0 (double) : Initial Stock Price.
                    mu (double) :  Drift.
                    sigma (double) : Volatility.
                    T (double) : Length of time in years.
                    n_steps (int) : Number of time steps.

            Returns:
                    S (double): Array of Stock Prices.
    '''
    Z = np.zeros( n_steps+1 )  # Initialise Z vector with (n_steps+1) entries all zeros.
    dt = T/n_steps  # Time interval size.
    Z[0] = np.log(S0)
    
    epsilon = np.random.randn( n_steps)  # epsilon vector of Standard Normal values.
    for i in range(0,n_steps):
        Z[i+1] = Z[i] + (mu - 0.5*sigma**2) * dt + sigma*np.sqrt(dt)*epsilon[i]
        
    S = np.exp(Z)
    return S

def simulate_gbm_paths( S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulate discrete time geometric Brownian motion paths. Returns a matrix of stock price paths. Each row in the matrix is     a different path and each column represents a different time point like a plot: time goes horizontally, different paths     are aligned vertically.
   
            Parameters:
                    S0 (double) : Initial Stock Price.
                    mu (double) :  Drift.
                    sigma (double) : Volatility.
                    T (double) : Length of time in years.
                    n_steps (int) : Number of time steps.
                    n_paths (int) : Number of Stock Price Paths to simulate.

            Returns:
                    S (double): Matrix of Stock Price Paths.
                    times (double) : Array of time points.
    
    """
    Z = np.zeros( [ n_paths, n_steps+1] )  # Matrix of (n_paths X n_steps+1).
    dt = T/n_steps
    Z[:,0] = np.log(S0)  # First column.
    times = np.linspace(0,T,n_steps+1)
    
    epsilon = np.random.randn( n_paths, n_steps )
    for i in range(0,n_steps):
        Z[:,i+1] = Z[:,i] + (mu-0.5*sigma**2) * dt + sigma*np.sqrt(dt)*epsilon[:,i]
        
    S = np.exp(Z)
    return S, times

def historic_var(returns, weights, confidence_level=0.95):
    '''
    This function takes in a numpy array of stock returns where each column represents returns for a different
    stock. It then calculates the portfolio returns and computes the VaR based on the specified confidence 
    level (default is 95%). It also calculates VaR for individual stock.  
    
            Parameters:
                    returns (double) : 2D-Numpy array of Stock Price returns.
                    weights (double) : 1D-Numpy array of weights of our portfolio.
                    confidence_level(double) : Confidence level of VaR (default is 95%).
                 
            Returns:
                    portfolio_var (double): VaR of the portfolio at the confidence level.
                    individual_vars (np.array): Numpy array containing VaR for each stock.
    '''
    portfolio_returns = returns @ weights    
    portfolio_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    individual_vars = np.percentile(returns, (1 - confidence_level) * 100, axis=0)
    return portfolio_var, individual_vars

def historic_cvar(returns, weights, confidence_level=0.95):
    '''
    Also known as Expected Shortfall, is the expected loss given that the loss exceeds the VaR. It computes the 
    portfolio returns, calculates the portfolio VaR at the specified confidence level, and then computes the CVaR by taking
    the mean of the returns that fall below the VaR. It also calculates CVaR for individual stock. 
    
            Parameters:
                    returns (double) : 2D-Numpy array of Stock Price returns.
                    weights (double) : 1D-Numpy array of weights of our portfolio.
                    confidence_level(double) : Confidence level of CVaR.
                 
            Returns:
                    portfolio_cvar (double): CVaR of the portfolio at the confidence level. 
                    individual_cvars (np.array): Numpy array containing CVaR for each stock.
    '''
    portfolio_returns = returns @ weights
    portfolio_var,individual_vars = historic_var(returns, weights, confidence_level = confidence_level)
    
    portfolio_cvar = np.mean(portfolio_returns[portfolio_returns <= portfolio_var])
    individual_cvars = np.mean(returns[returns <= individual_vars.reshape(1, -1)], axis=0)
    return portfolio_cvar, individual_cvars

def parametric_var(portfolioReturn, portfolioStd, confidence_level=0.95):
    '''
    This function uses the mean and standard deviation of the portfolio returns to estimate the VaR based on 
    the normal distribution assumption. 
    
            Parameters:
                    portfolioReturn (double) : Mean returns of a portfolio.
                    portfolioStd (double) : Standard deviation of portfolio.
                    confidence_level (float) : Confidence level for VaR calculation (default is 0.95).

            Returns:
                    var (double) : Value at Risk at the specified confidence level.
    '''
    var = portfolioReturn - portfolioStd * norm.ppf(0.95)
    return -var

def parametric_cvar(portfolioReturn, portfolioStd, confidence_level=0.95):
    '''
    This function uses the mean and standard deviation of the portfolio returns to estimate the CVaR based on 
    the normal distribution assumption. 
    
            Parameters:
                    portfolioReturn (double) : Mean returns of a portfolio.
                    portfolioStd (double) : Standard deviation of portfolio.
                    confidence_level (float) : Confidence level for VaR calculation (default is 0.95).

            Returns:
                    cvar (double) : CVaR of the portfolio at the confidence level. 
    '''
    cvar = portfolioStd * (1-confidence_level)**-1 * norm.pdf(norm.ppf(1-confidence_level)) - portfolioReturn
    return cvar

def blackscholes(r, S, K, T, sigma, t=0, type="call"):
    '''
    Calculates the Black-Scholes price of a European call/put option with strike K and maturity T.
    
            Parameters:
                    r (double) : Risk-free Interest Rate.
                    S (double) : Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    T (double) : Time to expiration (in years).
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    d1 = (np.log(S/K) + (r + sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    try:
        if type == "call":
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*(T-t))*norm.cdf(d2, 0, 1)
        elif type == "put":
            price = K*np.exp(-r*(T-t))*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'call' for Call or 'put' for Put.")

def call_payoff(S, K):
    '''
    Calculates the payoff of a European Call option with strike K.
    
            Parameters:
                    S (double) : Current price of the underlying asset.
                    K (double) : Strike price of the option.

            Returns:
                    payoff (double): Payoff of a European Call option.
    '''
    payoff = np.maximum(S - K,0)
    return payoff

def put_payoff(S, K):
    '''
    Calculates the payoff of a European Put option with strike K.
    
            Parameters:
                    S (double) : Current price of the underlying asset.
                    K (double) : Strike price of the option.

            Returns:
                    payoff (double): Payoff of a European Put option.
    '''
    payoff = np.maximum(K - S,0)
    return payoff

def vega(S, K, T, r, sigma):
    '''
    Calculates the greek Vega of a European call/put option.
    
            Parameters:
                    r (double) : Risk-free Interest Rate.
                    S (double) : Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    T (double) : Time to expiration (in years).
                    sigma (double) : Volatility of the underlying asset.

            Returns:
                    vega (double): Greek Vega of a European call/put option.
    '''
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    assert vega != 0, "vega is zero here."
    return vega

def realised_vol(ticker, start, end, interval):
    """ 
    Calculates Realised/Historical Volatility of Assets and prints it in a table.
                    
            Parameters:
                    ticker (string) : List of Stock tickers.
                    start (datetime) :  Start date of stock data to be extracted.
                    end (datetime) :  End date of stock data to be extracted.
                    interval (string) : Interval of stock data to be extracted (“1d”, “1wk”, “1mo”).

            Returns:
                    vol_interval (double) : 1D Numpy array of either daily, weekly or monthly 
                                            Volatility for assets depending on parameter interval.
                    vol (double) : 1D Numpy array of Annualized Volatility of Assets.
    """
    if interval == '1d':
        returns_data = read_return_history(ticker, start, end, interval = interval)[0]
        vol_interval = np.std(returns_data, axis = 0)*100 
        vol = vol_interval * np.sqrt(252)
    elif interval == '1wk':
        returns_data = read_return_history(ticker, start, end, interval = interval)[0]
        vol_interval = np.std(returns_data, axis = 0)*100 
        vol = vol_interval * np.sqrt(52)
    else:
        returns_data = read_return_history(ticker, start, end, interval = interval)[0]
        vol_interval = np.std(returns_data, axis = 0)*100 
        vol = vol_interval * np.sqrt(12)
    
    # Displaying Data in a table.
    data = {
    'Ticker': ticker,
    'Volatility': vol_interval,
    'Volatility Annualized': vol}
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    return vol_interval,vol

def implied_vol(r, S, K, T, market_price, type = 'call', tol=0.001, method = 'Newton-Raphson'):
    '''
    Compute the implied volatility of a European Option.
    
            Parameters:
                    r (double) : Risk-free Interest Rate.
                    S (double) : Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    T (double) : Time to expiration (in years).
                    market_price (double) : Market observed price.
                    type (String) : European call or put option.
                    tol (double) : Tolerance.
                    method () : Root finding algorithm (default is 'Newton-Raphson' ) 

            Returns:
                    implied_vol (double): Implied volatility of a European Option.
    '''
    max_iter = 200 # Maximum number of iterations.
    sigma_old = 0.50 # Initial guess.
    
    if method == 'Newton-Raphson':
        for k in range(max_iter):
            print(f'Number of iteration : {k}')
            blackscholes_price = blackscholes(r, S, K, T, sigma_old, type = type)  # BS Price.
            Cprime =  vega(S, K, T, r, sigma_old)    # Vega.
            C = blackscholes_price - market_price
            sigma_new = sigma_old - C/Cprime    # Newton-Raphson formula.
            blackscholes_price_new = blackscholes(r, S, K, T, sigma_new, type = type)

            if (abs(sigma_old - sigma_new) < tol or abs(blackscholes_price_new - market_price) < tol):
                break
            sigma_old = sigma_new

        implied_vol = sigma_new
        return implied_vol
    
    elif method == 'secant':
        def f( sigma ):
            return blackscholes(r, S, K, T, sigma, type = type) - market_price
        sol = scipy.optimize.root_scalar(f, x0=0.01, x1=1.0, method='secant')
        implied_vol = sol.root
        assert sol.converged
        return implied_vol   
    
    elif method == 'bisect':
        def f( sigma ):
            return blackscholes(r, S, K, T, sigma, type = type) - market_price
        sol = scipy.optimize.root_scalar(f, bracket=[0.3, 0.56], method='bisect')
        implied_vol = sol.root
        assert sol.converged
        return implied_vol  
    
    elif method == 'newton':
        def f( sigma ):
            return blackscholes(r, S, K, T, sigma, type = type) - market_price
        def fprime(sigma):
            return vega(S, K, T, r, sigma_old)
        sol = scipy.optimize.root_scalar(f, x0=0.15, fprime = fprime, method='newton')
        implied_vol = sol.root
        assert sol.converged
        return implied_vol   
    else:
        print("Please choose correct method")

def test_compute_implied_volatility():
    S0 = 100
    K = 110
    T = 0.5
    r = 0.02
    sigma = 0.8
    V =  blackscholes(r, S0, K, T, sigma, t=0, type="call")
    sigma_implied = implied_vol(r, S0, K, T, V, type = 'call', tol=0.001, method = 'Newton-Raphson')
    np.testing.assert_almost_equal(sigma,sigma_implied)
        
def one_step_wiener(T):
    """Generate a discrete time Wiener process on [0,T]
       with one step"""
    W1 = np.random.randn(1)*np.sqrt(T)
    return np.array([0,W1[0]]),np.array([0,T])

def riffle( v1, v2 ):
    """Take two vectors and alternate their entries
       to create a new vector"""    
    assert len(v1)==len(v2)+1
    ret = np.zeros(2*len(v1)-1)
    for i in range(0,len(v1)):
        ret[2*i] = v1[i]
    for i in range(0,len(v2)):
        ret[2*i+1] = v2[i]    
    return ret

def test_riffle():
    v1 = np.array([0,1,2])
    v2 = np.array([0.5,1.5])
    v = riffle(v1,v2)
    expected = np.array([0,0.5,1,1.5,2])
    assert np.linalg.norm(v-expected) < 0.001

def compute_halfway_times(t):
    """Compute the times halfway between those in the vector t"""
    all_but_last=t[0:-1]
    all_but_first=t[1:]
    return 0.5*(all_but_last + all_but_first)

def test_compute_halfway_times():
    t = np.array([0,1/4,1/2,3/4,1])
    halfway_times = compute_halfway_times(t)
    expected = np.array([1/8,3/8,5/8,7/8])
    for i in range(0,len(expected)):
        assert abs(halfway_times[i]-expected[i])<0.001    

def simulate_intermediate_values(W, t):    
    """Simulate values of a Wiener process at the halfway times of t given the values at t"""
    delta_t = t[1:]-t[0:-1]
    halfway_values = 0.5*(W[1:]+W[0:-1])
    eps = np.random.randn(len(W)-1)
    return halfway_values + 0.5*np.sqrt(delta_t)*eps

def test_simulate_intermediate_values():
    np.random.seed(1) # Always use the same "random" numbers
    n_samples = 10000
    d1 = np.zeros(n_samples)
    d2 = np.zeros(n_samples)
    for i in range(0,n_samples):
        W,t = one_step_wiener(1)
        intermediate = simulate_intermediate_values(W,t)
        W_1 = W[1]
        W_0 = W[0]
        W_half = intermediate[0]
        d1[i] = W_half-W_0
        d2[i] = W_1-W_half
    cov = np.cov(d1,d2)
    expected = np.array([[0.5,0],[0,0.5]])
    assert np.linalg.norm(cov-expected)<0.05

def wiener_interpolate( W, t, repeats=1 ):
    """Take a discrete time Wiener process at times given
       by the vector t and generate an interpolated process at
       the halfway times. Repeat this interpolation a number of times"""
    for i in range(0,repeats):
        halfway_times = compute_halfway_times(t)
        halfway_values = simulate_intermediate_values(W, t)
        W = riffle(W,halfway_values)
        t = riffle(t,halfway_times)
    return W,t

def N(x):
    """Calculates CDF of Standard Normal Distribution."""
    return norm.cdf(x)

def compute_d1_and_d2( S, t, K, T, r, sigma):
    """Computes d1 and d2 in the formula for Black-Scholes European Call Option."""
    tau = T-t
    d1 = 1/(sigma*np.sqrt(tau))*(np.log(S/K) + (r+0.5*sigma**2)*tau)
    d2 = d1 - sigma*np.sqrt(tau)
    return d1,d2

def black_scholes_call_delta(S,t,K,T,r,sigma):
    """Computes Delta of European Call Option."""
    d1, d2 = compute_d1_and_d2(S,t,K,T,r,sigma)
    return N(d1)

def black_scholes_put_delta(S,t,K,T,r,sigma):
    """Computes Delta of European Put Option."""
    d1, d2 = compute_d1_and_d2(S,t,K,T,r,sigma)
    return N(d1)-1

def black_scholes_call_gamma(S, t, K, T, r):
    "Calculate gamma of a option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    gamma = norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(tau))
    return gamma

def black_scholes_call_vega(S, t, K, T, r):
    "Calculate vega of a option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    vega = 0.01 *S*norm.pdf(d1, 0, 1)*np.sqrt(tau)  # 1% change in sigma.
    return vega

def black_scholes_call_theta(S, t, K, T, r):
    "Calculate theta of a call option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    theta = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(tau)) - r*K*np.exp(-r*tau)*norm.cdf(d2, 0, 1)
    theta = theta/365    # time decay per day.
    return theta

def black_scholes_put_theta(S, t, K, T, r):
    "Calculate theta of a put option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    theta = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(tau)) + r*K*np.exp(-r*tau)*norm.cdf(-d2, 0, 1)
    theta = theta/365
    return theta

def black_scholes_call_rho(S, t, K, T, r):
    "Calculate rho of a call option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    rho = K*tau*np.exp(-r*tau)*norm.cdf(d2, 0, 1)
    return 0.01*rho    # 1% change in Interest rate.

def black_scholes_put_rho(S, t, K, T, r):
    "Calculate rho of a put option"
    tau = T-t
    d1 = (np.log(S/K) + (r + sigma**2/2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    rho = -K*tau*np.exp(-r*tau)*norm.cdf(-d2, 0, 1)
    return 0.01*rho

# Binomial Option Pricing.
def binomial_pricer(S0, K, r, T, M, sigma, type = "call"):
    '''
    Calculates the Price of a European call/put option with strike K and maturity T using Binomial Model. 
    Uses Cox-Ross-Rubinstein model.
    
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u     # Down-factor in Binomial model. 1/u ensures that it is a recombining tree.
    p = (np.exp(r*dt) - d) / (u-d)    # Risk-Neutral probability measure.     
    S  = np.zeros( (M, M) )    # Initialising Stock Price matrix with all zeros.
    O = np.zeros( (M, M) )     # Initialising Option Price matrix with all zeros.

    # Generating Stock Prices.
    S[0, 0] = S0
    #  Fill out the remaining values.
    for i in range(1, M ):
        Q = i + 1
        S[i, 0] = d * S[i-1, 0]
        for j in range(1, M ):
            S[i, j] = u * S[i - 1, j - 1]
 
    #  Calculate the option price at expiration.
    if type == "call":
        expiration = S[-1,:] - K
    elif type == "put":
        expiration = K - S[-1,:]
           
    expiration.shape = (expiration.size, )
    expiration = np.where(expiration >= 0, expiration, 0)
    O[-1,:] =  expiration   # Set the last row of the Options matrix to our expiration values.

    #  Backpropagate to fill the remaining Options values at each nodes.
    for i in range(M - 2,-1,-1):
        for j in range(i + 1):
            O[i,j] = np.exp(-r * dt) * ((1-p) * O[i+1,j] + p * O[i+1,j+1])

    return O[0,0]

def binomial_pricer_vectorized(S0, K, r, T, M, sigma, type = "call"):
    '''
    Calculates the Price of a European call/put option with strike K and maturity T using Binomial Model.
    Uses Cox-Ross-Rubinstein model.
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u     # Down-factor in Binomial model. 1/u ensures that it is a recombining tree.
    p = (np.exp(r*dt) - d) / (u-d)    # Risk-Neutral probability measure. 
   
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    

    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N,0,-1):
        O = np.exp(-r*dt) * ( p * O[1:i+1] + (1-p) * O[0:i] )

    return O[0]

def binomial_pricer_vectorized_JR(S0, K, r, T, M, sigma, type = "call"):
    '''
    Calculates the Price of a European call/put option with strike K and maturity T using Binomial Model.
    Uses Jarrow and Rudd (JR) Method.
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    u = np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt))    # Jarrow and Rudd u parameter.
    d = np.exp((r - 0.5*sigma**2)*dt - sigma*np.sqrt(dt))    # Jarrow and Rudd d parameter.
    p = 0.5    # Jarrow and Rudd p parameter which is equal Risk-Neutral probability measure. 
   
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    

    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N,0,-1):
        O = np.exp(-r*dt) * ( p * O[1:i+1] + (1-p) * O[0:i] )

    return O[0]

def binomial_pricer_vectorized_EQP(S0, K, r, T, M, sigma, type = "call"):
    '''
    Calculates the Price of a European call/put option with strike K and maturity T using Binomial Model.
    Uses Equal Probabilities (EQP) Method.
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    nu = r - 0.5*sigma**2
    delta_xu = 0.5*nu*dt + 0.5*np.sqrt(4*sigma**2 * dt - 3*nu**2 * dt**2)  # EQP dxu parameter.
    delta_xd = 1.5*nu*dt - 0.5*np.sqrt(4*sigma**2 * dt - 3*nu**2 * dt**2)  # EQP dxd parameter.
    pu = 0.5  # EQP pu parameter.(probability of delta x up).
       
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * np.exp(N*delta_xd) * np.exp(np.arange(0, N+1, 1) * (delta_xu - delta_xd))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    

    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N,0,-1):
        O = np.exp(-r*dt) * ( pu * O[1:i+1] + (1-pu) * O[0:i] )

    return O[0]

def binomial_pricer_vectorized_TRG(S0, K, r, T, M, sigma, type = "call"):
    '''
    Calculates the Price of a European call/put option with strike K and maturity T using Binomial Model.
    Uses Trigeorgis Method(TRG).
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : European call or put option.

            Returns:
                    price (double): Price of a European call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    nu = r - 0.5*sigma**2
    delta_xu = np.sqrt(sigma**2 * dt + nu**2 * dt**2)  # TRG dxu parameter.
    delta_xd = -delta_xu  # TRG dxd parameter.
    pu = 0.5 + 0.5*nu*dt/delta_xu  # TRG pu parameter.(probability of delta x up).
       
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * np.exp(N*delta_xd) * np.exp(np.arange(0, N+1, 1) * (delta_xu - delta_xd))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    

    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N,0,-1):
        O = np.exp(-r*dt) * ( pu * O[1:i+1] + (1-pu) * O[0:i] )

    return O[0]

def binomial_pricer_upandout(S0, K, r, T, M, sigma, B, type = "call"):
    '''
    Calculates the Price of a Barrier call/put option with strike K and maturity T using Binomial Model.
    Uses Cox-Ross-Rubinstein model.
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    B (double) : Barrier Value.
                    type (String) : Barrier call or put option.

            Returns:
                    price (double): Price of a Barrier call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u     # Down-factor in Binomial model. 1/u ensures that it is a recombining tree.
    p = (np.exp(r*dt) - d) / (u-d)    # Risk-Neutral probability measure. 
   
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    
        
    # Checking if Barrier is crossed at Maturity.
    O[S >= B] = 0

    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        O[:i+1] = np.exp(-r*dt) * ( p * O[1:i+2] + (1-p) * O[0:i+1] )
        O = O[:-1]
        O[S >= B] = 0

    return O[0]

def binomial_pricer_American(S0, K, r, T, M, sigma, type = "put"):
    '''
    Calculates the Price of a American call/put option with strike K and maturity T using Binomial Model.
    Uses Cox-Ross-Rubinstein model.
            Parameters:
                    S0 (double): Current price of the underlying asset.
                    K (double) : Strike price of the option.
                    r (double) : Risk-free Interest Rate.
                    T (double) : Time to maturity/expiration (in years).
                    M (Int)    : Number of points in time direction.
                    sigma (double) : Volatility of the underlying asset.
                    type (String) : American call or put option.

            Returns:
                    price (double): Price of a American call/put option.
    '''
    N = M-1     # Number of time steps.
    dt = T/N    # Time-step size.
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u     # Down-factor in Binomial model. 1/u ensures that it is a recombining tree.
    p = (np.exp(r*dt) - d) / (u-d)    # Risk-Neutral probability measure. 
   
    # Initialising Asset prices at maturity - Time step N.
    S = S0 * d ** (np.arange(N,-1,-1)) * u ** (np.arange(0,N+1,1))

    # Initialise option values at maturity.
    if type == "call":
        O = np.maximum( S - K , np.zeros(N+1) )
    elif type == "put":
        O = np.maximum( K - S , np.zeros(N+1) )    
    
    # Backpropagate to fill the remaining Options values at each nodes.
    for i in np.arange(N-1,-1,-1):
        S = S0 * d**(np.arange(i,-1,-1)) * u**(np.arange(0,i+1,1))
        O[:i+1] = np.exp(-r*dt) * ( p * O[1:i+2] + (1-p) * O[0:i+1] )
        O = O[:-1]
        if type == "call":
            O = np.maximum( S - K , O )
        elif type == "put":
            O = np.maximum( K - S , O )    

    return O[0]

# Finite Difference Method To Calculate Option Prices.
def compute_vector_abc( K, T, sigma, r, S, dt, dS ):
    """S should be a vector containing all the stock points"""
    a = -sigma**2 * S**2/(2* dS**2 ) + r*S/(2*dS)  # Vector containing all values of a for i = 1,2,...,(M-1).
    b = r + sigma**2 * S**2/(dS**2)  # Vector containing all values of b for i = 1,2,...,(M-1).
    c = -sigma**2 * S**2/(2* dS**2 ) - r*S/(2*dS)  # Vector containing all values of c for i = 1,2,...,(M-1).
    return a,b,c

def compute_matrix_lambda( a,b,c ):
    """a[1:] is the Lower diagonal, b is the Main diagonal and c[:-1] is the Upper diagonal. The parameter offsets
    sets position of each diagonals."""
    return sc.diags( [a[1:],b,c[:-1]],offsets=[-1,0,1],format='csr')

def compute_vector_W(a,b,c, V0, VM):
    M = len(b)+1
    W = np.zeros(M-1)
    W[0] = a[0]*V0
    W[-1] = c[-1]*VM
    return W

def bottom_boundary_condition_call( K, T, S_min, r, t):
    """t should be a vector containing all the time points.
       This returns a vector of prices corresponding to the time points."""
    return np.zeros(t.shape)

def top_boundary_condition_call( K, T, S_max, r, t):
    """t should be a vector containing all the time points.
       This returns a vector of prices corresponding to the time points."""
    return S_max-np.exp(-r*(T-t))*K

def terminal_condition_call( K, T, S ):
    """S should be a vector containing all the final prices.
       This returns a vector of payoffs corresponding to the final stock prices"""
    return np.maximum(S-K,0)

def pricing_call_explicit( K, T, r, sigma, N, M):
    """Return values V is Matrix. S and t are vectors."""
    # Initialising the shape of the grid.
    dt = T/N
    S_min=0
    S_max=K*np.exp(8*sigma*np.sqrt(T))  # Should be s.t. we are unlikely to hit K. Hence, need Smax to be 8 x STD.DEV away from K.
    dS = (S_max-S_min)/M  # delta S.
    S = np.linspace(S_min,S_max,M+1)  # Vector of Stock Prices.
    t = np.linspace(0,T,N+1)  # Vector of times.
    V = np.zeros((N+1,M+1)) # Initialising option price matrix with all element zero.
    
    # Setting the Boundary Conditions.
    V[:,-1] = top_boundary_condition_call(K,T,S_max,r,t)
    V[:,0] = bottom_boundary_condition_call(K,T,S_max,r,t)
    V[-1,:] = terminal_condition_call(K,T,S)
    
    # Apply the recurrence relation.
    a,b,c = compute_vector_abc(K,T,sigma,r,S[1:-1],dt,dS)  # Computing vector a,b and c.
    Lambda = compute_matrix_lambda( a,b,c)  # Computing matrix lambda.
    identity = sc.identity(M-1, format='csr')  # Identity matrix.
    
    for i in range(N,0,-1):  # Looping backwards in time.
        W = compute_vector_W(a,b,c,V[i,0],V[i,M])
        # Use `dot` to multiply a vector by a sparse matrix
        V[i-1,1:M] = (identity-Lambda*dt).dot( V[i,1:M] ) - W*dt
        
    return V, t, S

def plot_option_price(V,t,S):
    # We only plot the points for the first 1/3 of the stock prices so that the detail is visible.
    M = len(S)-1
    S_zoom = S[0:int(M/3)]
    V_zoom=V[:,0:int(M/3)]
    t_mesh, S_mesh = np.meshgrid(t,S_zoom)
    ax = plt.axes(projection='3d')
    ax.plot_surface(t_mesh,S_mesh,V_zoom.T, alpha=0.8,edgecolor='#da07ed');
    ax.set_xlabel('Time (t)', fontsize = 12)
    ax.set_ylabel('Stock price (S)', fontsize = 12)
    ax.set_zlabel('Option price', fontsize = 12)
    
def pricing_call_implicit( K, T, r, sigma, N, M):
    """Return values V is Matrix. S and t are vectors."""
    # Initialising the shape of the grid.
    dt = T/N
    S_min=0
    S_max=K*np.exp(8*sigma*np.sqrt(T))  # Should be s.t. we are unlikely to hit K. Hence, need Smax to be 8 x STD.DEV away from K.
    dS = (S_max-S_min)/M  # delta S.
    S = np.linspace(S_min,S_max,M+1)  # Vector of Stock Prices.
    t = np.linspace(0,T,N+1)  # Vector of times.
    V = np.zeros((N+1,M+1)) # Initialising option price matrix with all element zero.
    
    # Setting the Boundary Conditions.
    V[:,-1] = top_boundary_condition_call(K,T,S_max,r,t)
    V[:,0] = bottom_boundary_condition_call(K,T,S_max,r,t)
    V[-1,:] = terminal_condition_call(K,T,S)
    
    # Apply the recurrence relation.
    a,b,c = compute_vector_abc(K,T,sigma,r,S[1:-1],dt,dS)  # Computing vector a,b and c.
    Lambda = compute_matrix_lambda( a,b,c)  # Computing matrix lambda.
    identity = sc.identity(M-1, format='csr')  # Identity matrix.
          
    for i in range(N-1,-1,-1):  # Looping backwards in time.
        W = compute_vector_W(a,b,c,V[i,0],V[i,M])
        V[i,1:M] = sc.linalg.spsolve(
            identity+Lambda*dt,  # A.
            V[i+1,1:M] - W*dt)   # b. in Ax=b.
        
    return V, t, S

def pricing_call_crank_nicolson( K, T, r, sigma, N, M):
    """Return values V is Matrix. S and t are vectors."""
    # Initialising the shape of the grid.
    dt = T/N
    S_min=0
    S_max=K*np.exp(8*sigma*np.sqrt(T))  # Should be s.t. we are unlikely to hit K. Hence, need Smax to be 8 x STD.DEV away from K.
    dS = (S_max-S_min)/M  # delta S.
    S = np.linspace(S_min,S_max,M+1)  # Vector of Stock Prices.
    t = np.linspace(0,T,N+1)  # Vector of times.
    V = np.zeros((N+1,M+1)) # Initialising option price matrix with all element zero.
    
    # Setting the Boundary Conditions.
    V[:,-1] = top_boundary_condition_call(K,T,S_max,r,t)
    V[:,0] = bottom_boundary_condition_call(K,T,S_max,r,t)
    V[-1,:] = terminal_condition_call(K,T,S)
    
    # Apply the recurrence relation.
    a,b,c = compute_vector_abc(K,T,sigma,r,S[1:-1],dt,dS)  # Computing vector a,b and c.
    Lambda = compute_matrix_lambda( a,b,c)  # Computing matrix lambda.
    identity = sc.identity(M-1, format='csr')  # Identity matrix.
          
    for i in range(N-1,-1,-1):  # Looping backwards in time.
        Wt = compute_vector_W(a,b,c,V[i,0],V[i,M])
        Wt_plus_dt = compute_vector_W(a,b,c,V[i+1,0],V[i+1,M])
        V[i,1:M] = sc.linalg.spsolve(
            identity+0.5*Lambda*dt,  # A.
            (identity-0.5*Lambda*dt).dot(V[i+1,1:M]) - 0.5*dt*(Wt_plus_dt + Wt))   # b. in Ax=b.
        
    return V, t, S

# Monte Carlo Option Pricing.
def monte_carlo_pricer( S0, r, sigma, T, n_steps, n_paths, payoff_function ):
    """Returns Price and Confidence Interval."""
    S_twiddle, times = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths )
    payoffs = payoff_function(S_twiddle)  # Payoff Vector.
    p = 99
    alpha = scipy.stats.norm.ppf((1-p/100)/2)
    price = np.exp(-r*T)*np.mean( payoffs )  # Expectation of discounted payoff.
    sigma_sample = np.exp(-r*T) * np.std( payoffs )
    lower = price + alpha*sigma_sample/np.sqrt(n_paths)
    upper = price - alpha*sigma_sample/np.sqrt(n_paths)
    return  price, lower, upper

def asian_call_payoff( S, K ):
    """Returns Payoff Vector."""
    S_bar = np.mean(S,axis=1)  # S_bar is average over time points i.e. rows so axis = 1.
    return np.maximum( S_bar-K, 0 )

def price_asian_call_monte_carlo( S0, r, sigma, K, T, n_steps, n_paths ):
    def payoff_fn(S):
        return asian_call_payoff(S,K)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn )

def digital_call_payoff(S, K):
    """Returns Payoff Vector of ones or zeros."""
    S_T = S[:,-1]  # Final value of Stock price at time T.
    payoff = S_T>K
    return payoff

def price_digital_call_monte_carlo(S0, r, sigma, K, T, n_steps, n_paths):
    def payoff_fn(S):
        return digital_call_payoff(S,K)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn) 

def digital_call_price_black_scholes(S, r, sigma, K, T, t=0):
    """Analytical Price of Digital Call using Black Scholes formula."""
    d1,d2 = compute_d1_and_d2( S, t, K, T, r, sigma)
    price = np.exp(-r*T) * scipy.stats.norm.cdf(d2)
    return price

# Testing with Vanilla Call.
def price_call_monte_carlo(S0, r, sigma, K, T, n_steps, n_paths):
    # Defining the payoff function for European Call. 
    def payoff_fn( S ):
        """ Takes an array of stock price as input and outputs payoff. """
        S_T = S[:,-1]
        payoff = np.maximum(S_T-K, 0)
        return payoff
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn)

def test_price_call_monte_carlo():
    np.random.seed(0)
    # Only one step is needed to price a Call option, hence n_steps = 1.
    price,low,high = price_call_monte_carlo(S0, r,sigma,K, T,1, n_paths)  # Price and Confidence Interval.
    true_price = blackscholes(r, S0, K, T, sigma, t=0, type="call")  # Actual price using Black Scholes Analytical formula.
    error = np.abs(true_price - price)
    # Checking whether Analytical BS price lies within Confidence Interval. 
    assert low<true_price
    assert true_price<high
    assert error<0.1
        
# Testing with Vanilla Put.
def price_put_monte_carlo( S0, r, sigma, K, T, n_steps, n_paths ):
    # Defining the payoff function for European Put. 
    def payoff_fn( S ):
        """ Takes an array of stock price as input and outputs payoff. """
        S_T = S[:,-1]
        payoff = np.maximum( K-S_T, 0 )
        return payoff
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn)

def test_price_put_monte_carlo():
    np.random.seed(0)
    # Only one step is needed to price a Put option, hence n_steps = 1.
    price,low,high = price_put_monte_carlo(S0, r,sigma,K, T,1, n_paths)  # Price and Confidence Interval.
    true_price = blackscholes(r, S0, K, T, sigma, t=0, type="put")  # Actual price using Black Scholes Analytical formula.
    error = np.abs(true_price - price) 
    # Checking whether Analytical BS price lies within Confidence Interval. 
    assert low<true_price
    assert true_price<high
    assert error<0.1

def test_asian_call_payoff():
    np.random.seed(0)
    S = np.array([[3,7,2],[1,2,3],[3,4,2],[1,1,1]])
    payoffs = asian_call_payoff(S,2)    # S_bar -K.
    expected = np.array([2,0,1,0])
    # Using function from numpy testing module. This function is particularly useful when dealing with floating-point numbers 
    # where exact equality checks might fail due to numerical imprecision.
    np.testing.assert_almost_equal( payoffs, expected, decimal=7)
    
def test_digital_call_price_black_scholes():
    np.random.seed(0)
    price = digital_call_price_black_scholes(S0,r,sigma,K,T)
    expected = np.exp(-r*T) * scipy.stats.norm.cdf(1/(sigma*np.sqrt(T))*(np.log(S0/K)+(r-0.5*sigma**2)*T))
    np.testing.assert_almost_equal( price, expected, decimal=7)
    
def test_price_digital_call_monte_carlo():
    np.random.seed(0)
    # Only one step is needed to price a Digital Call, hence n_steps = 1.
    price,low,high = price_digital_call_monte_carlo(S0, r,sigma,K, T,1, n_paths)  # Price and Confidence Interval.
    true_price = digital_call_price_black_scholes(S0,r,sigma,K,T)  # Actual price using Black Scholes Analytical formula.
    error = np.abs(true_price - price) 
    # Checking whether Analytical BS price lies within Confidence Interval. 
    assert low<true_price
    assert true_price<high
    assert error<0.1
    
def up_and_out_knockout_call_payoff(S, K, B):
    """Returns Payoff Vector."""
    S_T = S[:,-1]  # Final value of Stock price at time T. 
    highest_price = np.max(S,axis=1)
    payoff = np.where(highest_price >= B, 0, np.maximum(S_T - K, 0))
    return payoff

def price_up_and_out_knockout_call_monte_carlo(S0, r, sigma, K, B, T, n_steps, n_paths):
    def payoff_fn(S):
        return up_and_out_knockout_call_payoff(S,K,B)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn) 

def test_up_and_out_knockout_call_payoff():
    np.random.seed(0)
    S = np.array([[95, 100, 105, 108, 110, 107, 109],[102, 105, 95, 100, 108, 107, 108]])
    payoffs = up_and_out_knockout_call_payoff(S,K=102,B=109)   
    expected = np.array([0,6])
    np.testing.assert_almost_equal( payoffs, expected, decimal=7)
    
def down_and_out_knockout_call_payoff(S, K, B):
    """Returns Payoff Vector."""
    S_T = S[:,-1]  # Final value of Stock price at time T. 
    lowest_price = np.min(S,axis=1)
    payoff = np.where(lowest_price <= B, 0, np.maximum(S_T - K, 0))
    return payoff

def price_down_and_out_knockout_call_monte_carlo(S0, r, sigma, K, B, T, n_steps, n_paths):
    def payoff_fn(S):
        return down_and_out_knockout_call_payoff(S,K,B)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn) 

def test_down_and_out_knockout_call_payoff():
    np.random.seed(0)
    S = np.array([[95, 100, 105, 108, 110, 107, 109],[102, 105, 101, 100, 108, 107, 108]])
    payoffs = down_and_out_knockout_call_payoff(S,K=105,B=94)   
    expected = np.array([4,3])
    np.testing.assert_almost_equal( payoffs, expected, decimal=7)
    
def up_and_in_knockin_call_payoff(S, K, B):
    """Returns Payoff Vector."""
    S_T = S[:,-1]  # Final value of Stock price at time T. 
    highest_price = np.max(S,axis=1)
    payoff = np.where(highest_price < B, 0, np.maximum(S_T - K, 0))
    return payoff

def price_up_and_in_knockin_call_monte_carlo(S0, r, sigma, K, B, T, n_steps, n_paths):
    def payoff_fn(S):
        return up_and_in_knockin_call_payoff(S,K,B)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn) 

def test_up_and_in_knockin_call_payoff():
    np.random.seed(0)
    S = np.array([[95, 100, 105, 108, 110, 107, 109],[102, 105, 95, 100, 108, 107, 108]])
    payoffs = up_and_in_knockin_call_payoff(S,K=102,B=109)   
    expected = np.array([7,0])
    np.testing.assert_almost_equal( payoffs, expected, decimal=7)
    
def one_touch_payoff(S, B):
    """Returns Payoff Vector of ones or zeros."""
    highest_price = np.max(S,axis=1)
    payoff = np.where(highest_price >= B, 1, 0)
    return payoff

def price_one_touch_monte_carlo(S0, r, sigma, B, T, n_steps, n_paths):
    def payoff_fn(S):
        return one_touch_payoff(S, B)
    return monte_carlo_pricer(S0, r, sigma, T, n_steps, n_paths, payoff_fn) 

def test_one_touch_payoff():
    np.random.seed(0)
    S = np.array([[95, 100, 105, 108, 110, 107, 109],[102, 105, 95, 100, 108, 107, 108]])
    payoffs = one_touch_payoff(S, B=110)
    expected = np.array([1,0])
    np.testing.assert_almost_equal( payoffs, expected, decimal=7)
    
def delta_monte_carlo( S0, r, sigma, T, n_steps, n_paths, payoff_function ):
    h = S0*10**(-5)
    S_twiddle, times = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths )
    paths_for_S_minus_h = S_twiddle/S0*(S0-h)
    paths_for_S_plus_h = S_twiddle/S0*(S0+h)
    payoffs_S_plus_h = np.exp(-r*T)*payoff_function(paths_for_S_plus_h)
    payoffs_S_minus_h = np.exp(-r*T)*payoff_function(paths_for_S_minus_h)
    samples = (payoffs_S_plus_h-payoffs_S_minus_h)/(2*h)
    p = 99
    alpha = scipy.stats.norm.ppf((1-p/100)/2)
    delta = np.mean( samples )  # Computes Delta.
    sigma_sample = np.std( samples )  # Standard deviation of Delta.
    lower = delta + alpha*sigma_sample/np.sqrt(n_paths)
    upper = delta - alpha*sigma_sample/np.sqrt(n_paths)
    return lower, upper

def delta_call_confidence_interval_monte_carlo( S0, r, sigma, K, T, n_paths ):
    def call_payoff( S ):
        S_T = S[:,-1]
        return np.maximum( S_T-K, 0 )
    return delta_monte_carlo(S0, r, sigma, T, 1, n_paths, call_payoff )

def delta_call_monte_carlo( S0, r, sigma, K, T, n_paths ):
    low, high = delta_call_confidence_interval_monte_carlo(S0, r, sigma, K, T, n_paths )
    return 0.5*(low+high)  # Taking Average of the Confidence Interval.

def test_delta_call_monte_carlo():
    np.random.seed(0)
    K = 105; T = 1;
    S0 = 100; r = 0.05; sigma = 0.25
    n_paths = 100000;
    low,high = delta_call_confidence_interval_monte_carlo(S0, r,sigma,K, T, n_paths)
    bs_delta = black_scholes_call_delta(S0,0,K,T,r,sigma)
    assert low<bs_delta
    assert bs_delta<high
    
# Calibration And Jump-Diffusion Model.
def days_to_maturity(maturity_date):
    future_date = datetime.strptime(maturity_date, '%Y-%m-%d')
    today = datetime.today()
    diff = (future_date - today).days + 1
    return diff

def stock_price(ticker):
    symbol = ticker
    today = datetime.today().strftime('%Y-%m-%d')
    today_datetime = datetime.strptime(today, '%Y-%m-%d')
    start_date = (today_datetime - timedelta(days=3)).strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=today)
    stock_price = data['Adj Close'].iloc[-1]
    return stock_price

def get_rf_rate(maturity_days):
    # Ticker symbols for 13-week and 5-Year U.S. Treasury bills and Notes.
    symbol_11w = '^IRX'  # 13 WEEK TREASURY BILL.
    symbol_5yrs = '^FVX' # Treasury Yield 5 Years.
    
    today = datetime.today().strftime('%Y-%m-%d')
    today_datetime = datetime.strptime(today, '%Y-%m-%d')
    start_date = (today_datetime - timedelta(days=4)).strftime('%Y-%m-%d')
    data_13w = yf.download(symbol_11w, start = start_date, end = today)
    data_5yrs = yf.download(symbol_5yrs, start = start_date, end = today)
    rf_rate_13w = data_13w['Adj Close'].iloc[-1] / 100
    rf_rate_5yrs = data_5yrs['Adj Close'].iloc[-1] / 100
    
    maturities = [91, 1825]
    rates = [rf_rate_13w, rf_rate_5yrs]

    # Perform cubic spline interpolation.
    cs = CubicSpline(maturities, rates)
    riskfree_rate = cs(maturity_days)

    return riskfree_rate

def jump_diffusion_call_price( S, K, T, r, sigma, lbda, j):
    S = np.array(S)
    term = np.ones(S.shape)
    total = np.zeros(S.shape)
    n = 0
    mu_tilde = r + lbda*(1-j)
    while np.any(abs(term)>1e-7*abs(total)):
        V = blackscholes(mu_tilde, S, j**(-n)*K, T, sigma, t=0, type="call")   # r=mu, S=S, K=j**(-n)*K, T=T, sigma=sigma, t=0.
        term = ((j*lbda*T)**n)/np.math.factorial(n) * np.exp( -j*lbda*T) * V
        total = total + term
        n = n+1
    return total

def simulate_jump_diffusion( S0, T, mu_twiddle, sigma, lbda, j, n_steps, n_paths ):
    t = np.linspace(0,T,n_steps+1)
    dt = T/n_steps
    Z = np.zeros((n_paths,n_steps+1))
    Z[:,0]=np.log(S0)
    for i in range(0, n_steps):
        epsilon = np.random.normal(size=(n_paths))
        jumps = np.random.poisson(lbda*dt, n_paths )
        Z[:,i+1]=Z[:,i] + (mu_twiddle-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*epsilon + np.log(j)*jumps
    S = np.exp(Z)
    return S, t

def price_by_monte_carlo_jd( S0, r, sigma, lbda, j, T, n_steps,n_paths, payoff_function):
    mu_twiddle = r + lbda*(1-j)
    S,t = simulate_jump_diffusion( S0, T, mu_twiddle, sigma, lbda, j, n_steps, n_paths )
    payoffs = payoff_function(S)
    p = 99
    alpha = scipy.stats.norm.ppf((1-p/100)/2)
    price = np.exp(-r*T)*np.mean( payoffs )
    sigma_sample = np.exp(-r*T) * np.std( payoffs )
    lower = price + alpha*sigma_sample/np.sqrt(n_paths)
    upper = price - alpha*sigma_sample/np.sqrt(n_paths)
    return lower, upper

def price_call_by_monte_carlo_jd( S0, K, T, r, sigma, lbda, j, n_steps,n_paths ):
    # Define the payoff function, it takes an array. 
    def call_payoff( S ):
        S_T = S[:,-1]
        return np.maximum( S_T-K, 0 )
    return price_by_monte_carlo_jd(S0, r, sigma, lbda, j, T, n_steps, n_paths, call_payoff )

def test_price_call_by_monte_carlo_jd():
    np.random.seed(0)
    # Only one step is needed to price a call option.
    n_steps = 1
    K = S0
    low,high = price_call_by_monte_carlo_jd(S0, K, T, r,sigma,lbda,j, n_steps, n_paths)
    expected = jump_diffusion_call_price(S0,K,T,r,sigma, lbda, j)
    assert low<expected
    assert expected<high
    
def price_asian_call_by_monte_carlo_jd( S0, r, sigma, lbda, j, K, T, n_steps, n_paths ):
    def payoff_fn(S):
        return asian_call_payoff(S,K)
    return price_by_monte_carlo_jd(S0, r, sigma, lbda,j, T, n_steps, n_paths, payoff_fn )

def compute_jump_diffusion_ivols( S0, r, sigma, lbda, j, strikes, T ):  
    ''' Computes Implied Volatilities under Jump-Diffusion Model. 
        strikes = vector.
    '''
    implied_vols = np.zeros( len( strikes ))
    for i in range(0, len(strikes)):
        K = strikes[i]
        price = jump_diffusion_call_price(S0, K, T, r, sigma, lbda, j)
        implied_vols[i] = implied_vol(r, S0, K, T, price, type = 'call', tol=0.01, method = 'secant')
    return implied_vols

def plot_model_fit( sigma, lbda, j ):
    model_ivols = compute_jump_diffusion_ivols(S0, r, sigma, lbda, j, strikes, T)
    ax = plt.gca()
    ax.scatter( strikes, implied_vols, label='Market' );
    ax.plot( strikes, model_ivols, label='Model' );
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied volatility')
    ax.set_ylim(0,0.3);
    ax.set_xlabel('Strike')
    ax.set_ylabel('Implied volatility');
    ax.set_title('Implied volatilities');
    ax.legend()
    
def jump_diffusion_delta( S, K, time_to_maturity, r, sigma, lbda, j):
    S = np.array(S)
    term = np.ones(S.shape)
    total = np.zeros(S.shape)
    n = 0
    mu_tilde = r + lbda*(1-j)
    while np.any(abs(term)>1e-7*abs(total)):
        V = black_scholes_call_delta(S,0, j**(-n)*K,time_to_maturity,mu_tilde, sigma) #  (S=S,t=0,K=j**(-n)*K,T=time_to_maturity,r=mu_tilde,sigma)
        term = ((j*lbda*time_to_maturity)**n)/np.math.factorial(n) * np.exp( -j*lbda*time_to_maturity) * V
        total = total + term
        n = n+1
    return total

def test_jump_diffusion_delta():
    S0 = np.array([100,110])
    h = S0*10**(-5)
    K = 110
    T = 0.5
    r = 0.02
    sigma = 0.2          
    lbda = 1
    j = 0.9
    central_estimate = (jump_diffusion_call_price(S0+h,K,T,r,sigma,lbda,j) - jump_diffusion_call_price(S0-h,K,T,r,sigma,lbda,j))/(2*h)
    actual = jump_diffusion_delta( S0, K, T, r, sigma, lbda, j)
    np.testing.assert_almost_equal( central_estimate, actual, decimal=4 )
    
def solve_two_by_two( a,b,c,d, v1, v2):
    det = a*d-b*c
    alpha = (d*v1 - b*v2)/det
    beta = (-c*v1 + a*v2)/det
    return alpha,beta

def test_solve_two_by_two():
    a = np.array([1,2])
    b = np.array([2,-1])
    c = np.array([3,1])
    d = np.array([3,2])
    v1 = np.array([4,5])
    v2 = np.array([3,2])
    x,y = solve_two_by_two(a,b,c,d,v1,v2)
    np.testing.assert_almost_equal( a*x + b*y, v1 )
    np.testing.assert_almost_equal( c*x + d*y, v2 )
    
def compute_alpha_and_beta(S, r, sigma, lbda, j, K_prime, K, time_to_maturity ):
    a = 1 
    b = jump_diffusion_delta(S,K_prime, time_to_maturity, r,sigma, lbda, j)
    v1 = jump_diffusion_delta(S,K, time_to_maturity, r,sigma, lbda, j)
    c = S - j*S
    d = jump_diffusion_call_price( S, K_prime, time_to_maturity, r, sigma, lbda, j )- \
            jump_diffusion_call_price( j*S, K_prime, time_to_maturity, r, sigma, lbda, j )
    v2 = jump_diffusion_call_price( S, K, time_to_maturity, r, sigma, lbda, j )- \
            jump_diffusion_call_price( j*S, K, time_to_maturity, r, sigma, lbda, j )
    alpha,beta = solve_two_by_two(a,b,c,d,v1,v2)
    return alpha, beta

def simulate_replication( S0, r, sigma, lbda, j, K_prime, K, T, lambda_prime, mu, n_steps, n_paths):
    S, t = simulate_jump_diffusion( S0, T, mu, sigma, lbda, j, n_steps, n_paths )
    dt = T/n_steps
    wealth = jump_diffusion_call_price( S[:,0], K, T, r, sigma, lbda, j)
    for i in range(0, n_steps):
        time_to_maturity = T - t[i]
        stock_price = S[:,i]
        hedging_option_price = jump_diffusion_call_price(stock_price, K_prime, time_to_maturity, r, sigma, lbda, j )
        alpha, beta = compute_alpha_and_beta(stock_price, r, sigma, lbda, j, K_prime, K, time_to_maturity )
        bank = wealth - alpha*stock_price - beta*hedging_option_price
        new_bank = np.exp(r*dt)*bank
        new_stock_price = S[:,i+1]
        new_option_price = jump_diffusion_call_price(new_stock_price, K_prime, time_to_maturity-dt, r, sigma, lbda, j )
        wealth = new_bank + alpha*new_stock_price + beta*new_option_price
    ST = S[:,-1]
    return ST, wealth


