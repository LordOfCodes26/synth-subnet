import numpy as np
import pandas as pd
import requests
import bittensor as bt
from datetime import datetime
from arch import arch_model
from properscoring import crps_ensemble


def get_asset_price(asset="BTC"):
    """
    Retrieves the current price of the specified asset.
    Currently, supports BTC via Pyth Network.

    Returns:
        float: Current asset price.
    """
    if asset == "BTC":
        btc_price_id = (
            "0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
        )
        endpoint = f"https://hermes.pyth.network/v2/updates/price/stream?ids[]={btc_price_id}"  # TODO: this endpoint is deprecated
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()[0]  # First item in the list
            
            price = float(data["price"]["price"]) * (10 ** int(data["price"]["expo"]))
            return price

        except Exception as e:
            print(f"Error: {e}")
            return None
    else:
        # For other assets, implement accordingly
        print(f"Asset '{asset}' not supported.")
        return None


# def simulate_single_price_path(
#     current_price, time_increment, time_length, sigma
# ):
#     """
#     Simulate a single crypto asset price path.
#     """
#     one_hour = 3600
#     dt = time_increment / one_hour
#     num_steps = int(time_length / time_increment)
#     std_dev = sigma * np.sqrt(dt)
#     price_change_pcts = np.random.normal(0, std_dev, size=num_steps)
#     cumulative_returns = np.cumprod(1 + price_change_pcts)
#     cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
#     price_path = current_price * cumulative_returns
#     return price_path


# def simulate_crypto_price_paths(
#     current_price, time_increment, time_length, num_simulations, sigma
# ):
#     """
#     Simulate multiple crypto asset price paths.
#     """

#     price_paths = []
#     for _ in range(num_simulations):
#         price_path = simulate_single_price_path(
#             current_price, time_increment, time_length, sigma
#         )
#         price_paths.append(price_path)

#     return np.array(price_paths)


def get_SVJD_parameters(start_time, time_increment: int, time_length: int, asset="Crypto.BTC/USD") -> dict:

    start_time = datetime.fromisoformat(start_time).timestamp()
    # Define the Pyth TradingView endpoint for historical BTC data
    pyth_tv_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

    # Set parameters for data retrieval (adjust the start and end times)
    params = {
        "symbol": asset,
        "from": int(start_time - time_increment * (time_length // time_increment - 1)),  
        "to": int(start_time),
        "resolution": f"{time_increment // 60}" #calculate minute
    }

    # Fetch data from Pyth API
    response = requests.get(pyth_tv_url, params=params)
    data = response.json()

    # Convert response to DataFrame
    btc_df = pd.DataFrame({
        "timestamp": data["t"],  # Time in Unix format
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    })
    # Convert timestamp to readable datetime format
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], unit="s")

    # Compute log returns for SVJD calibration
    btc_df["log_return"] = np.log(btc_df["close"] / btc_df["close"].shift(1))

    # Drop NaN values
    btc_df.dropna(inplace=True)

    # Compute annualized drift (mean return)
    mu = btc_df["log_return"].mean() * time_length / time_increment  # 
    # Compute initial variance (V0)
    V0 = btc_df["log_return"].var()

    # Define jump threshold (2 standard deviations)
    threshold = 2 * btc_df["log_return"].std()

    # Identify jumps as large price movements
    jumps = btc_df[np.abs(btc_df["log_return"]) > threshold]["log_return"]

    # Compute jump intensity (λ): Average jumps per day
    lambda_jump = len(jumps) / len(btc_df) * time_length / time_increment  

    # Mean jump size (μ_J)
    mu_J = jumps.mean()

    # Jump size volatility (σ_J)
    sigma_J = jumps.std()


    # Fit GARCH(1,1) model to log returns
    garch_model = arch_model(btc_df["log_return"], vol="Garch", p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")

    # Extract GARCH model parameters
    kappa = garch_fit.params["omega"]  # Mean reversion speed
    theta = garch_fit.conditional_volatility.mean()**2  # Long-run variance
    sigma_V = garch_fit.params["alpha[1]"]  # Volatility of variance

    # Compute rolling volatility (10-period moving standard deviation)
    btc_df["rolling_vol"] = btc_df["log_return"].rolling(window=10).std()

    # Estimate correlation between returns and volatility
    rho = btc_df["log_return"].corr(btc_df["rolling_vol"])

    svjd_params = {
        "mu": mu,
        "V0": V0,
        "kappa": kappa,
        "theta": theta,
        "sigma_V": sigma_V,
        "rho": rho,
        "lambda_jump": lambda_jump,
        "mu_J": mu_J,
        "sigma_J": sigma_J
    }
    return svjd_params



def simulate_crypto_price_paths_SVID(current_price, start_time, time_increment, time_length, num_simulations) -> np.array:
    SVID_params = get_SVJD_parameters(start_time=start_time, time_increment=time_increment, time_length=time_length)
    bt.logging.info(
            f"SVID_params: {SVID_params}"
        )
    S0 = current_price
    T = time_length
    N = time_length // time_increment
    dt = N/T
    mu = SVID_params["mu"] # Expected return
    V0 = SVID_params["V0"] # Initial variance
    kappa = SVID_params['kappa'] # Mean reversion speed of variance
    theta = SVID_params['theta'] # Long-run variance
    sigma_V = SVID_params['sigma_V'] # Volatility of variance
    rho = SVID_params['rho'] # Correlation between BTC returns and variance
    lambda_jump = SVID_params['lambda_jump'] # Jump intensity (expected jumps per year)
    mu_J = SVID_params['mu_J'] # Average jump size (log-normal)
    sigma_J = SVID_params['sigma_J'] # Jump volatility
    num_paths = num_simulations
    
    # Correlated Brownian motions
    W_S = np.random.randn(N, num_paths)  # BTC price Brownian motion
    W_V = rho * W_S + np.sqrt(1 - rho**2) * np.random.randn(N, num_paths)  # Volatility Brownian motion

    # Initialize price and variance paths
    S = np.zeros((N, num_paths))
    V = np.zeros((N, num_paths))
    S[0, :] = S0
    V[0, :] = V0

    # Simulate paths
    for t in range(1, N):
        # Stochastic variance process (Heston-like)
        V[t] = np.maximum(V[t-1] + kappa * (theta - V[t-1]) * dt + sigma_V * np.sqrt(V[t-1] * dt) * W_V[t], 0)
        
        # Jump component (Poisson process)
        num_jumps = np.random.poisson(lambda_jump * dt, num_paths)  # 0 or 1 most of the time
        Jumps = np.where(num_jumps > 0, np.exp(mu_J + sigma_J * np.random.randn(num_paths)) - 1, 0)  # Only apply when jump occurs
        # BTC price process
        S[t] = S[t-1] * np.exp((mu - 0.5 * V[t]) * dt + np.sqrt(V[t] * dt) * W_S[t]) * (1 + Jumps)

    bt.logging.info(
            f"S: {S}"
        )
        
    return np.transpose(S)
