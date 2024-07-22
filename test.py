import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

def black_scholes_call_price_vectorized(S, K, T, r, sigma):
    """
    Vectorized Black-Scholes call option price calculation.

    Parameters:
    S (float): Spot price of the underlying asset.
    K (np.array): Array of strike prices.
    T (float): Time to maturity (in years).
    r (float): Risk-free interest rate.
    sigma (np.array): Array of volatilities.

    
    Returns:
    np.array: Array of call option prices.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_prices

def get_volatility_smile(strikes, implied_vols, num_points=5000):
    """
    Function to get the volatility smile using cubic spline interpolation.

    Parameters:
    strikes (list or np.array): Array of strike prices.
    implied_vols (list or np.array): Array of implied volatilities corresponding to the strike prices.
    num_points (int): Number of points for the interpolated strike range.

    Returns:
    strike_range (np.array): Interpolated range of strike prices.
    fitted_vols (np.array): Fitted volatilities over the interpolated strike range.
    """
    # Fit a cubic spline to the implied volatility data
    spline_fit = CubicSpline(strikes, implied_vols, bc_type='clamped')

    # Generate a range of strikes for the spline
    strike_range = np.linspace(min(strikes), max(strikes), num_points)
    fitted_vols = spline_fit(strike_range)

    return strike_range, fitted_vols

# Example usage
if __name__ == "__main__":
    # Example data based on the provided context
    spot_price = 4000  # Example spot price of the index
    risk_free_rate = 0.01  # Example risk-free rate
    time_to_maturity = 0.25  # Example 3 months (in years)
    
    # Moneyness ratios and implied volatilities from the provided text
    moneyness = np.array([80.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0, 120.0])
    implied_vols = np.array([23.95, 21.71, 18.81, 17.40, 16.09, 14.88, 13.84, 12.48, 12.34])  # Convert to decimals
    
    # Convert moneyness to strike prices
    strikes = spot_price * (moneyness / 100)

    # Get the volatility smile
    strike_range, fitted_vols = get_volatility_smile(strikes, implied_vols)

    # Calculate call prices using the vectorized Black-Scholes formula
    call_prices = black_scholes_call_price_vectorized(spot_price, strike_range, time_to_maturity, risk_free_rate, fitted_vols)

    # Plot the original and smoothed implied volatilities
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, implied_vols, 'o', label='Original Implied Vols')
    plt.plot(strike_range, fitted_vols, '-', label='Fitted Cubic Spline')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('Implied Volatility Smile')
    plt.legend()
    plt.show()

    # Plot call prices vs. strike prices
    plt.figure(figsize=(10, 6))
    plt.plot(strike_range, call_prices, label='Call Prices')
    plt.xlabel('Strike Price')
    plt.ylabel('Call Price')
    plt.title('Call Prices vs Strike Prices')
    plt.legend()
    plt.show()

    # Calculate second derivative of call prices with respect to strike prices
    second_derivative = np.gradient(np.gradient(call_prices, strike_range), strike_range)

    # Apply Savitzky-Golay filter to smooth the second derivative
    second_derivative_smooth = savgol_filter(second_derivative, window_length=51, polyorder=3)

    # Plot second derivatives
    plt.figure(figsize=(10, 6))
    plt.plot(strike_range, second_derivative, label='Second Derivative of Call Prices')
    plt.plot(strike_range, second_derivative_smooth, label='Smoothed Second Derivative', linestyle='--')
    plt.xlabel('Strike Price')
    plt.ylabel('Second Derivative')
    plt.title('Second Derivative of Call Prices vs Strike Prices')
    plt.legend()
    plt.show()

    # Calculate RND using Breeden-Litzenberger formula
    rnd = np.exp(risk_free_rate * time_to_maturity) * second_derivative_smooth

    # Ensure non-negative densities
    rnd = np.maximum(rnd, 0)

    # Plot RND
    plt.figure(figsize=(10, 6))
    plt.plot(strike_range, rnd, label='Risk-Neutral Density')
    plt.xlabel('Strike Price')
    plt.ylabel('Density')
    plt.title('Risk-Neutral Density of the Index')
    plt.legend()
    plt.show()