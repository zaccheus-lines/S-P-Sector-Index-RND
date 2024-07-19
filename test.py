import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

# Example data from Bloomberg Terminal
spot_price = 4000  # Example spot price of the index
risk_free_rate = 0.01  # Example risk-free rate
time_to_maturity = 0.5  # Example 6 months

data = {
    'strike': [3800, 3900, 4000, 4100, 4200, 4300, 4400],
    'implied_vol': [0.21, 0.19, 0.18, 0.19, 0.20, 0.22, 0.24],
    'delta': [0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95]  # Given delta values
}

option_data = pd.DataFrame(data)

# Fit a smoothing spline to the implied volatility data in delta space
lambda_param = 6
spline_fit = UnivariateSpline(option_data['delta'], option_data['implied_vol'], s=lambda_param)

# Generate a range of deltas for plotting the spline
delta_range = np.linspace(min(option_data['delta']), max(option_data['delta']), 5000)
fitted_vols = spline_fit(delta_range)

# Plot the original and smoothed implied volatilities
plt.figure(figsize=(10, 6))
plt.plot(option_data['delta'], option_data['implied_vol'], 'o', label='Original Implied Vols')
plt.plot(delta_range, fitted_vols, '-', label='Smoothed Spline')
plt.xlabel('Delta')
plt.ylabel('Implied Volatility')
plt.title('Implied Volatility Smoothing')
plt.legend()
plt.show()

# Black-Scholes call price function
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Convert back to strike space
strike_prices = np.zeros(len(delta_range))
call_prices = np.zeros(len(delta_range))

for i, delta in enumerate(delta_range):
    vol = fitted_vols[i]
    d1 = norm.ppf(delta) if delta < 0.5 else norm.ppf(1 - delta)
    strike_prices[i] = spot_price * np.exp(-d1 * vol * np.sqrt(time_to_maturity))
    call_prices[i] = black_scholes_call_price(spot_price, strike_prices[i], time_to_maturity, risk_free_rate, vol)

# Use numpy arrays
strike_prices = np.array(strike_prices)
call_prices = np.array(call_prices)

# Calculate second derivative of call prices with respect to strike prices
second_derivative = np.gradient(np.gradient(call_prices, strike_prices), strike_prices)

# Calculate RND using Breeden-Litzenberger formula
rnd = np.exp(risk_free_rate * time_to_maturity) * second_derivative

# Ensure non-negative densities
rnd = np.maximum(rnd, 0)

# Plot RND
plt.figure(figsize=(10, 6))
plt.plot(strike_prices, rnd, label='Risk-Neutral Density')
plt.xlabel('Strike Price')
plt.ylabel('Density')
plt.title('Risk-Neutral Density of the Index')
plt.legend()
plt.show()