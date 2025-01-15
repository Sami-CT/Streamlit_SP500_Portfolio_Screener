import pandas as pd
import yfinance as yf

constituents = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/_r/-/data/constituents.csv")
constituents["Symbol"] = constituents["Symbol"].str.replace(r"\.", "-", regex=True)
cn = len(constituents)

def api_request(a, b, c, d):
    x = yf.Ticker(a)
    y = x.history(start=b, end=c, interval=d)
    return y

data = pd.DataFrame()

print("Téléchargement des données du S&P 500...")
sp500 = api_request("^GSPC", "2019-11-01", "2024-11-01", "1d")
sp500 = sp500.reset_index()
sp500 = sp500[["Date", "Close"]].rename(columns={"Close": "SP500 Close"})

for i, row in constituents.iterrows():
    symbol = row["Symbol"]
    security = row["Security"]
    sector = row["GICS Sector"]
    industry = row["GICS Sub-Industry"]

    print(f"[{i + 1}/{cn}] {symbol}...", end="")
    history = api_request(symbol, "2019-11-01", "2024-11-01", "1d")
    print(" X")
    history = history.reset_index()

    history.insert(0, "Security", security)
    history["GICS Sector"] = sector
    history["GICS Sub-Industry"] = industry
    history["Symbol"] = symbol

    data = pd.concat([data, history])


print("Fusion des données avec le S&P 500...")
data["Date"] = pd.to_datetime(data["Date"])
sp500["Date"] = pd.to_datetime(sp500["Date"])
data = pd.merge(data, sp500, on="Date", how="left")

print()
print(50 * "*")
print("Données en cours d'exportation...")
data.to_csv("imported_data.csv", sep=";", index=False)
print("Données exportées avec succès !")
print(50 * "*")