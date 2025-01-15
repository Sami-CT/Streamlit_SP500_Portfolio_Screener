import pandas as pd
import numpy as np

data = pd.read_csv("imported_data.csv", sep=";")

data["Performance Intraday (%)"] = ((data["Close"] - data["Open"]) / data["Open"]) * 100
data["30-Day Moving Average (USD)"] = data["Close"].rolling(window=30).mean()


data["Asset Return"] = data.groupby("Symbol")["Close"].pct_change()  # Rendements de l'actif
data["Benchmark Return"] = data["SP500 Close"].pct_change()         # Rendements du S&P 500


data["Excess Return"] = data["Asset Return"] - data["Benchmark Return"]


def calculate_information_ratio(group):
    mean_excess_return = group["Excess Return"].mean()  # Moyenne des rendements excédentaires
    tracking_error = group["Excess Return"].std()       # Écart-type des rendements excédentaires
    if tracking_error == 0:
        return np.nan  # Éviter la division par zéro
    return mean_excess_return / tracking_error


information_ratios = data.groupby("Symbol").apply(calculate_information_ratio)
information_ratios.name = "Information Ratio"


data = data.merge(information_ratios, on="Symbol", how="left")

data["Asset Return (%)"] = data["Asset Return"] * 100
data["Benchmark Return (%)"] = data["Benchmark Return"] * 100
data["Excess Return (%)"] = data["Excess Return"] * 100
columns_to_drop = ["Asset Return", "Benchmark Return", "Excess Return", "Information Ratio_x", "Information Ratio_y"]
data = data.drop(columns=columns_to_drop, errors="ignore")

data["Asset Return"] = data["Asset Return (%)"] / 100  # Convert to fraction
data["Benchmark Return"] = data["Benchmark Return (%)"] / 100  # Convert to fraction

def calculate_rolling_beta(group):
    rolling_cov = group["Asset Return"].rolling(window=30).cov(group["Benchmark Return"])
    rolling_var = group["Benchmark Return"].rolling(window=30).var()
    return rolling_cov / rolling_var

data["Rolling Beta (30 days)"] = (
    data.groupby("Symbol", group_keys=False)
    .apply(calculate_rolling_beta)
)


def add_max_drawdown_30_days(data):

    column_returns = "Asset Return (%)"
    max_drawdown_column_name = "Max Drawdown (30 days)"
    window_size = 30

    if column_returns not in data.columns:
        raise ValueError(f"La colonne '{column_returns}' n'existe pas dans le DataFrame.")


    def calculate_max_drawdown_window(returns):
        if len(returns) < window_size: 
            return np.nan
        
        cumulative_returns = np.cumprod(1 + returns / 100) 
        
        drawdown = (cumulative_returns / np.maximum.accumulate(cumulative_returns)) - 1
        return drawdown.min()  # Drawdown le plus sévère


    try:
        data[max_drawdown_column_name] = (
            data.groupby("Symbol")[column_returns]
            .rolling(window=window_size)
            .apply(calculate_max_drawdown_window, raw=True)
            .reset_index(level=0, drop=True)
        )
    except Exception as e:
        print(f"Erreur lors du calcul du Max Drawdown : {e}")

    return data


def add_var_30_days(data):

    column_returns = "Asset Return (%)"
    var_column_name = "VaR (5%) (30 days)"
    window_size = 30
    confidence_level = 5

    if column_returns not in data.columns:
        raise ValueError(f"La colonne '{column_returns}' n'existe pas dans le DataFrame.")


    def calculate_var_window(returns):
        if len(returns) < window_size:
            return np.nan
        return np.percentile(returns, confidence_level)


    data[var_column_name] = (
        data.groupby("Symbol")[column_returns]
        .rolling(window=window_size)
        .apply(calculate_var_window, raw=True)
        .reset_index(level=0, drop=True)
    )

    return data

data = add_max_drawdown_30_days(data)
data = add_var_30_days(data)

def add_daily_ytd(data):

    data["Date"] = pd.to_datetime(data["Date"], utc=True)
    
    data = data.sort_values(["Symbol", "Date"])
    
    def calculate_daily_ytd(group):

        group["Year"] = group["Date"].dt.year
        first_close_per_year = group.groupby("Year")["Close"].transform("first")
        
        group["YTD Performance (%)"] = ((group["Close"] - first_close_per_year) / first_close_per_year) * 100
        
        if group["Year"].min() == data["Date"].dt.year.min():
            group.loc[group["Year"] == data["Date"].dt.year.min(), "YTD Performance (%)"] = np.nan
        
        return group

    data = data.groupby("Symbol", group_keys=False).apply(calculate_daily_ytd)
    data = data.drop(columns=["Year"])  # Clean up temporary column

    return data

    
data = add_daily_ytd(data)

def get_data():
    return data

data = data.rename(columns={
    "Symbol": "Ticker",
    "Open": "Price Open (USD)",
    "High": "Price High (USD)",
    "Low": "Price Low (USD)",
    "Close": "Price Close (USD)",
    "SP500 Close": "S&P500 Close (USD)"
    })

data = data.drop(columns=["Stock Splits"])
data = data.drop(columns=["Asset Return"])
data = data.drop(columns=["Benchmark Return"])

data["Price Range (USD)"] = data["Price High (USD)"] - data["Price Low (USD)"]
data["Price Change (USD)"] = data["Price Close (USD)"] - data["Price Open (USD)"]
data["Price Change (%)"] = (data["Price Close (USD)"] - data["Price Open (USD)"]) / data["Price Open (USD)"] * 100
data["Volatility (%)"] = (data["Price High (USD)"] - data["Price Low (USD)"]) / data["Price Low (USD)"] * 100
data["Dividend Yield (%)"] = data["Dividends"] / data["Price Close (USD)"] * 100
data["Average Price (USD)"] = (data["Price Open (USD)"] + data["Price Close (USD)"]) / 2
data["Volume to Price Ratio"] = data["Volume"] / data["Price Close (USD)"]

data["Date"] = pd.to_datetime(data["Date"], utc=True)
data["Date"] = data["Date"].dt.strftime('%Y-%m-%d')

data = data.sort_values(by=["Ticker", "Date"])
data = data.set_index(["Ticker", "Date"])

print()
print(50 * "*")
print("Données en cours d'exportation...")
data.to_csv("data.csv", sep=";", index=True)
print("Données exportées avec succès !")
print(50 * "*")