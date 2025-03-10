import numpy as np
import pandas as pd
import requests
from datetime import datetime
import bittensor as bt
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
        endpoint = f"https://hermes.pyth.network/api/latest_price_feeds?ids[]={btc_price_id}"  # TODO: this endpoint is deprecated
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()[0]  # First item in the list         

            price = float(data["price"]["price"]) * (10 ** int(data["price"]["expo"]))
            print(f"BTC Price: ${price}")

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


def get_Heston_parameters(start_time, time_increment: int, time_length: int, asset="Crypto.BTC/USD") -> dict:
    start_time = datetime.fromisoformat(start_time).timestamp()
    pyth_tv_url = "https://benchmarks.pyth.network/v1/shims/tradingview/history"

    params = {
        "symbol": asset,
        "from": int(start_time - time_increment * (time_length * 180 // time_increment - 1)),  
        "to": int(start_time),
        "resolution": f"{60}"  # Minute intervals
    }

    response = requests.get(pyth_tv_url, params=params)
    data = response.json()

    btc_df = pd.DataFrame({
        "timestamp": data["t"],
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"]
    })
    btc_df["timestamp"] = pd.to_datetime(btc_df["timestamp"], unit="s")
    btc_df["log_return"] = np.log(btc_df["close"] / btc_df["close"].shift(1))
    btc_df.dropna(inplace=True)

    # Drift
    mu = btc_df["log_return"].mean()
    # Initial variance
    V0 = btc_df["log_return"].var()

    # Fit GARCH(1,1) model to estimate volatility parameters
    garch_model = arch_model(btc_df["log_return"], vol="Garch", p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")

    kappa = garch_fit.params["omega"]
    theta = garch_fit.conditional_volatility.mean()**2
    sigma = garch_fit.params["alpha[1]"]

    btc_df["rolling_vol"] = btc_df["log_return"].rolling(window=10).std()
    rho = btc_df["log_return"].corr(btc_df["rolling_vol"])

    heston_params = {
        "mu": mu,
        "V0": V0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho
    }
    return heston_params



def simulate_crypto_price_paths_SVID(current_price, start_time, time_increment, time_length, num_simulations) -> np.array:
    heston_params = get_Heston_parameters(start_time=start_time, time_increment=time_increment, time_length=time_length)
    bt.logging.info(f"Here is SVID_params: {heston_params}")
    print(heston_params)

    S0 = current_price
    T = time_length / 86400  # Convert seconds to days
    N = time_length // time_increment  # Number of steps
    dt = T / N

    mu = heston_params["mu"]
    V0 = heston_params["V0"]
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    sigma = heston_params["sigma"]
    rho = heston_params["rho"]

    num_paths = num_simulations

    # Generate correlated Brownian motions
    W_S = np.random.randn(N, num_paths)
    W_V = rho * W_S + np.sqrt(1 - rho**2) * np.random.randn(N, num_paths)

    # Initialize price and variance paths
    S = np.zeros((N, num_paths))
    V = np.zeros((N, num_paths))
    S[0, :] = S0
    V[0, :] = V0

    # Simulate paths using the Euler-Maruyama method
    for t in range(1, N):
        V[t] = np.maximum(V[t-1] + kappa * (theta - V[t-1]) * dt + sigma * np.sqrt(V[t-1] * dt) * W_V[t], 0)
        S[t] = S[t-1] * np.exp((mu - 0.5 * V[t]) * dt + np.sqrt(V[t] * dt) * W_S[t])

    return np.transpose(S)
