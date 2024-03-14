import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import cvxopt
from scipy.stats import norm
import scipy.optimize

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