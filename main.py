import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.express as px
from unidecode import unidecode
from io import BytesIO
from FunctionMining import *
stopWords = [unidecode(sw) for sw in stopWords]
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('french')
from wordcloud import WordCloud
from FunctionMining import *

data = pd.read_csv("data_small.csv", sep=";")
graph_data = data
graph_data['Date'] = pd.to_datetime(graph_data['Date'])
graph_data['Year'] = graph_data['Date'].dt.year
graph_data['Month'] = graph_data['Date'].dt.month

def next_monday(date):
    days_ahead = 7 - date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return date + timedelta(days_ahead)

def display_wordcloud():
    st.title("Wordcloud")
    fig1, buffer1 = generate_wordcloud(Texte, "blue")
    st.write("**WordCloud 1**")
    st.pyplot(fig1)
    st.download_button("Télécharge notre wordcloud", buffer1, "wordcloud.png", mime='image/png')

    # Ajout de fichier pour permettre à l'utilisateur de générer son propre wordcloud et le télécharger
    file = st.file_uploader("Choisissez un fichier type txt pour le wordcloud")
    if file:
        try:
            contenu = file.read().decode("utf-8")
            file2 = stem_cleaner(contenu, stemmer, stopWords)
            fig2, buffer2 = generate_wordcloud(file2, "green")
        except Exception as e:
            st.markdown("")

    if file is not None:
        try:
            st.write("**WordCloud 2**")
            st.pyplot(fig2)
            st.download_button("Télécharge ton wordcloud", buffer2, "wordcloud.png", mime='image/png')
        except Exception as e:
            st.markdown("Il y a eu une erreur.")
            st.markdown("Avez vous bien uploader un fichier de type text")

def display_equity_screening():

    # ========================= FONCTIONS =========================

    def condition_match(a, b, c, d):
        if b == "==":
            y = d[a] == c
        elif b == ">":
            y = d[a] > c
        elif b == ">=":
            y = d[a] >= c
        elif b == "<":
            y = d[a] < c
        elif b == "<=":
            y = d[a] <= c
        else:
            y = d[a] != c
        return y
    
    def count_matches(a, b, c, d):
        y = condition_match(a, b, c, d)
        return d[y].shape[0]
    
    def add_condition_table(a, b, c, x, d):
        z = pd.DataFrame([{
            "Column": a,
            "Operator": b,
            "Value": c,
            "Matches": count_matches(a, b, c, d)
            }])
        y = pd.concat([x, z], ignore_index=True)
        return y
    
    def remove_condition_table(a, b, c, x):
        z = ((x["Column"] == a) & (x["Operator"] == b) & (x["Value"] == c))
        if z.any():
            y = x[~z]
        else:
            return x
        return y
    
    def selection_update(d):
        y = d
        if d.empty:
            y = pd.Dataframe()
            return y
        else:
            y = y.rename(columns={"Date": "Selected"})
            y["Selected"] = True
            y.set_index("Ticker", inplace=True)
        return y

    def results_update(x, d):
        y = d
        if x.empty:
            y = pd.DataFrame()
            return y
        else:
            for _, row in x.iterrows():
                a, b, c = row["Column"], row["Operator"], row["Value"]
                z = condition_match(a, b, c, y)
                y = y[z]
        return y
    
    def export_to_csv(r, s):
        z = r
        z.set_index("Ticker", inplace=True)
        y = z[s["Selected"] == True]
        y.to_csv("export.csv", index=True, sep=";")
        n = len(y)
        st.success(f"Succesfully exported informations on {n} securities.")

    # ========================= INTERFACE =========================

    st.title("Equity Screening")

    available_dates = pd.to_datetime(data["Date"].unique()).sort_values()
    default_date = datetime(2024, 10, 31)

    if default_date not in available_dates:
        default_date = available_dates[-1]

    if "selected_date" not in st.session_state:
        st.session_state.selected_date = default_date

    selected_date = st.date_input(
        "As of",
        value=st.session_state.selected_date,
        min_value=available_dates.min(),
        max_value=available_dates.max(),
        key="date_input"
    )

    if selected_date.strftime("%Y-%m-%d") not in data["Date"].values:
        st.warning("Selected date has no data. Moving to the next Monday.")
        st.session_state.selected_date = next_monday(selected_date)
        st.write(f"Updated date to: {st.session_state.selected_date.strftime('%Y-%m-%d')}")

    filtered_data = data[data["Date"] == st.session_state.selected_date.strftime("%Y-%m-%d")]

    if "results" not in st.session_state:
        st.session_state["results"] = pd.DataFrame()
    
    if "selection" not in st.session_state:
        st.session_state["selection"] = pd.DataFrame()

    if "condition_table" not in st.session_state:
        st.session_state["condition_table"] = pd.DataFrame(columns=["Column", "Operator", "Value", "Matches"])

    st.header("Screening Criterias")

    name_columns = ["Security", "Ticker"]
    industry_columns = ["GICS Sector", "GICS Sub-Industry"]
    market_data_columns = [col for col in filtered_data.columns if col not in ["Date"] + industry_columns + name_columns]

    if "selected_column" not in st.session_state:
        st.session_state.selected_column = "Ticker"
        st.session_state.selected_operator = "=="
        st.session_state.selected_value = ""

    criteria1, criteria2, criteria3 = st.columns(3)
    with criteria1:
        if st.button("Name"):
            st.session_state.last_clicked = "Name"
    with criteria2:
        if st.button("Industry"):
            st.session_state.last_clicked = "Industry"
    with criteria3:
        if st.button("Market Data"):
            st.session_state.last_clicked = "Market Data"

    if "last_clicked" in st.session_state:
        if st.session_state.last_clicked == "Name":
            filtered_name_columns = [col for col in name_columns if col in filtered_data.columns]
            st.session_state.selected_column = st.radio("Select Name Column:", filtered_name_columns)
        elif st.session_state.last_clicked == "Industry":
            filtered_industry_columns = [col for col in industry_columns if col in filtered_data.columns]
            st.session_state.selected_column = st.radio("Select Industry Column:", filtered_industry_columns)
        elif st.session_state.last_clicked == "Market Data":
            filtered_market_data_columns = [col for col in market_data_columns if col in filtered_data.columns]
            st.session_state.selected_column = st.radio("Select Market Data Column:", filtered_market_data_columns)

    st.header("Add Criterias")

    selected_column = st.session_state.selected_column
    st.text_input("Selected column:", value=selected_column, disabled=True)

    operators = {"numeric": ["==", "<", "<=", ">", ">=", "!="], "categorical": ["==", "!="], "name": ["==", "!"]}
    if selected_column in industry_columns:
        unique_values = filtered_data[selected_column].dropna().unique()
        selected_operator = st.selectbox("Select an operator:", options=operators["categorical"])
        selected_value = st.selectbox("Select a value:", options=unique_values)
    elif selected_column in name_columns:
        selected_operator = st.selectbox("Select an operator:", options=operators["name"])
        selected_value = st.text_input("Enter a name:")
    else:
        selected_operator = st.selectbox("Select an operator:", options=operators["numeric"])
        selected_value = st.text_input("Enter a numeric value:")
        if selected_value:
            try:
                selected_value = float(selected_value)
            except ValueError:
                st.error("Please enter a valid numeric value.")

    st.session_state.selected_operator = selected_operator
    st.session_state.selected_value = selected_value

    add_col, remove_col = st.columns([1, 1])

    with add_col:
        add_criteria = st.button("Add Criteria")
    with remove_col:
        remove_criteria = st.button("Remove Criteria")

    if add_criteria and selected_column and selected_operator and selected_value:
        st.session_state["condition_table"] = add_condition_table(
            selected_column,
            selected_operator,
            selected_value,
            st.session_state["condition_table"],
            filtered_data
            )
        
        st.session_state["results"] = results_update(
            st.session_state["condition_table"],
            filtered_data
            )
        st.session_state["selection"] = selection_update(st.session_state["results"])
        
        match_count = st.session_state["condition_table"]["Matches"].iloc[-1]
        st.success(f"Criteria added with {match_count} matches.")

    if remove_criteria and selected_column and selected_operator and selected_value:
        st.session_state["condition_table"] = remove_condition_table(
            selected_column,
            selected_operator,
            selected_value,
            st.session_state["condition_table"])

        st.session_state["results"] = results_update(st.session_state["condition_table"], filtered_data)
        st.session_state["selection"] = selection_update(st.session_state["results"])

        st.success("Criteria removed.")

    st.header("Selected Screening Criterias")

    if not st.session_state["condition_table"].empty:
        st.dataframe(st.session_state["condition_table"])
    else:
        st.write("No criteria added yet.")

    st.header(f"Results ({len(st.session_state['results'])})")

    if st.session_state["selection"].empty:
        st.write("No results.")
    else:
        updated_selection = st.data_editor(
            st.session_state["selection"],
            key="selection_editor"
        )
        if not updated_selection.equals(st.session_state["selection"]):
            st.session_state["selection"] = updated_selection

    no_results = st.session_state["selection"] is None or st.session_state["selection"].empty

    slct_all, selct_none, exp_csv = st.columns(3)

    with slct_all:
        select_all = st.button("Select All", disabled=no_results)
    with selct_none:
        select_none = st.button("Select None", disabled=no_results)
    with exp_csv:
        export_csv = st.button("Export Selection to CSV", disabled=no_results)

    if select_all:
        st.session_state["selection"]["Selected"] = True
        st.success("Selected all results.")
    if select_none:
        st.session_state["selection"]["Selected"] = False
        st.success("Deselected all results.")
    if export_csv:
        export_to_csv(st.session_state["results"], st.session_state["selection"])

def display_portfolio():
    # ========================= FONCTIONS =========================

    def portfolio_init(d):
        y = d
        if "Weight" not in y:
            z = pd.DataFrame()
            z["Ticker"] = y["Ticker"]
            z["Security"] = y["Security"]
            z["Position"] = 1000.0
            z = z.set_index("Ticker")
            y = z
        return y
    
    def load_portfolio(n):
        try:
            fn = f"{n}.csv"
            y = pd.read_csv(fn, sep=";")
            y = portfolio_init(y)
            st.session_state["portfolio"] = y
            update_portfolio_data()
            st.success(f"Successfully loaded portfolio: '{n}.csv'.")
        except FileNotFoundError:
            st.error(f"File '{n}.csv' not found.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")

    def save_portfolio(n, p):
        try:
            fn = f"{n}.csv"
            p.to_csv(fn, index=False, sep=";")
            st.success(f"Successfully saved portfolio as '{n}.csv'.")
        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")

    def input_load():
        st.session_state["input_visible"] = True

    def submit_file_name():
        file_name = st.session_state.get("input_text", "").strip()
        if file_name:
            st.session_state["file_name"] = file_name
            st.session_state["input_visible"] = False
            load_portfolio(file_name)
        else:
            st.error("Please enter a valid file name.")

    def input_name():
        st.session_state["input_name_visible"] = True

    def submit_portfolio_name():
        portfolio_name = st.session_state.get("input_portfolio_name", "").strip()
        if portfolio_name:
            st.session_state["portfolio_name"] = portfolio_name
            st.session_state["input_name_visible"] = False
        else:
            st.error("Please enter a valid portfolio name.")
    
    def update_portfolio_data():
        update_securities_global_data()
        update_securities_current_data()
        update_portfolio_global_data()
        update_portfolio_current_data()
        update_portfolio_stats()
    
    def update_securities_global_data():
        z = st.session_state["portfolio"].index.tolist()
        d = data
        st.session_state["securities_global_data"] = d[d["Ticker"].isin(z)]

    def update_securities_current_data():
        st.session_state["securities_current_data"] = st.session_state["securities_global_data"][st.session_state["securities_global_data"]["Date"] == st.session_state.selected_date.strftime("%Y-%m-%d")]

    def handle_date_change():
        st.session_state.selected_date = st.session_state.date_input
        update_portfolio_data()

    def update_portfolio_global_data():
        d = st.session_state["securities_global_data"]
        z = st.session_state["portfolio"]
        z = z.reset_index()
        z = z[["Ticker", "Position"]]
        d = d.merge(z, on="Ticker", how="left")
        d = d[d["Position"] > 0].copy()

        d["Market Value"] = d["Position"] * d["Price Close (USD)"]

        def weighted_average(column, group):
            return (group[column] * group["Position"]).sum() / group["Position"].sum()

        y = d.groupby("Date").apply(
            lambda group: pd.Series({
                "Total Position": group["Position"].sum(),
                "Total Market Value": group["Market Value"].sum(),
                "Price Open (USD)": weighted_average("Price Open (USD)", group),
                "Price High (USD)": weighted_average("Price High (USD)", group),
                "Price Low (USD)": weighted_average("Price Low (USD)", group),
                "Price Close (USD)": weighted_average("Price Close (USD)", group),
                "Volume": group["Volume"].sum(),
                "Dividends": group["Dividends"].sum(),
                "S&P500 Close (USD)": weighted_average("S&P500 Close (USD)", group),
                "Performance Intraday (%)": weighted_average("Performance Intraday (%)", group),
                "30-Day Moving Average (USD)": weighted_average("30-Day Moving Average (USD)", group),
                "Information Ratio": weighted_average("Information Ratio", group),
                "Asset Return (%)": weighted_average("Asset Return (%)", group),
                "Benchmark Return (%)": weighted_average("Benchmark Return (%)", group),
                "Excess Return (%)": weighted_average("Excess Return (%)", group),
                "Rolling Beta (30 days)": weighted_average("Rolling Beta (30 days)", group),
                "Max Drawdown (30 days)": weighted_average("Max Drawdown (30 days)", group),
                "VaR (5%) (30 days)": weighted_average("VaR (5%) (30 days)", group),
                "YTD Performance (%)": weighted_average("YTD Performance (%)", group),
                "Price Range (USD)": weighted_average("Price Range (USD)", group),
                "Price Change (USD)": weighted_average("Price Change (USD)", group),
                "Price Change (%)": weighted_average("Price Change (%)", group),
                "Volatility (%)": weighted_average("Volatility (%)", group),
                "Dividend Yield (%)": weighted_average("Dividend Yield (%)", group),
                "Average Price (USD)": weighted_average("Average Price (USD)", group),
                "Volume to Price Ratio": weighted_average("Volume to Price Ratio", group),
            })
        ).reset_index()
        st.session_state["portfolio_global_data"] = y
    
    def update_portfolio_current_data():
        st.session_state["portfolio_current_data"] = st.session_state["portfolio_global_data"][st.session_state["portfolio_global_data"]["Date"] == st.session_state.selected_date.strftime("%Y-%m-%d")]

    def update_portfolio_stats():
        z = st.session_state["portfolio_current_data"]
        x = ["Total Market Value",
              "Performance Intraday (%)",
              "Asset Return (%)",
              "Benchmark Return (%)",
              "Excess Return (%)",
              "Rolling Beta (30 days)",
              "Max Drawdown (30 days)",
              "VaR (5%) (30 days)",
              "YTD Performance (%)",
              "Volatility (%)"
              ]
        z = z[x]
        z = z.transpose()
        y = z.rename(columns={z.columns[0]: "Value"})
        st.session_state["portfolio_stats"] = y

    # ========================= INTERFACE =========================

    if "input_visible" not in st.session_state:
        st.session_state["input_visible"] = False
    if "file_name" not in st.session_state:
        st.session_state["file_name"] = ""
    if "input_name_visible" not in st.session_state:
        st.session_state["input_name_visible"] = False
    if "portfolio_name" not in st.session_state:
        st.session_state["portfolio_name"] = "Portfolio"
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = pd.DataFrame()
    if "securities_global_data" not in st.session_state:
        st.session_state["securities_global_data"] = pd.DataFrame()
    if "securities_current_data" not in st.session_state:
        st.session_state["securities_current_data"] = pd.DataFrame()
    if "portfolio_global_data" not in st.session_state:
        st.session_state["portfolio_global_data"] = pd.DataFrame()
    if "portfolio_current_data" not in st.session_state:
        st.session_state["portfolio_current_data"] = pd.DataFrame()
    if "portfolio_stats" not in st.session_state:
        st.session_state["portfolio_stats"] = pd.DataFrame()

    st.title(st.session_state["portfolio_name"])

    available_dates = pd.to_datetime(data["Date"].unique()).sort_values()
    default_date = datetime(2024, 10, 31)

    if default_date not in available_dates:
        default_date = available_dates[-1]

    if "selected_date" not in st.session_state:
        st.session_state.selected_date = default_date

    selected_date = st.date_input(
        "As of",
        value=st.session_state.selected_date,
        min_value=available_dates.min(),
        max_value=available_dates.max(),
        key="date_input",
        on_change=handle_date_change
    )

    if selected_date.strftime("%Y-%m-%d") not in data["Date"].values:
        st.warning("Selected date has no data. Moving to the next available date.")
        st.session_state.selected_date = available_dates.min()
        st.write(f"Updated date to: {st.session_state.selected_date.strftime('%Y-%m-%d')}")

    rename, load_port, save_port = st.columns(3)
    with rename:
        if st.button("Rename Portfolio"):
            input_name()
    with load_port:
        if st.button("Load Portfolio"):
            input_load()
    with save_port:
        if st.button("Save Portfolio"):
            if not st.session_state["portfolio"].empty:
                save_portfolio(st.session_state["portfolio_name"], st.session_state["portfolio"])
            else:
                st.error("Portfolio is empty. Nothing to save.")

    if st.session_state["input_visible"]:
        with st.form("file_name_form"):
            st.text_input("Enter file name:", key="input_text")
            if st.form_submit_button("Confirm"):
                submit_file_name()

    if st.session_state["input_name_visible"]:
        with st.form("portfolio_name_form"):
            st.text_input("Enter portfolio name:", key="input_portfolio_name")
            if st.form_submit_button("Confirm"):
                submit_portfolio_name()

    st.header("Portfolio Management")

    table1, table2 = st.columns(2)
    
    with table1:
        if not st.session_state["portfolio"].empty:
            st.subheader("Constituents")
            updated_portfolio = st.data_editor(
                st.session_state["portfolio"],
                num_rows="dynamic",
                key="portfolio_editor"
            )
            if not updated_portfolio.equals(st.session_state["portfolio"]):
                st.session_state["portfolio"] = updated_portfolio
                update_portfolio_data()
        else:
            st.subheader("Constituents")
            st.write("No portfolio loaded yet.")
    with table2:
        if not st.session_state["portfolio_stats"].empty:
            st.subheader("Informations")
            st.dataframe(st.session_state["portfolio_stats"])
        else:
            st.subheader("Informations")
            st.write("No data to show.")

def display_graphs():
    st.title("Graphs")
    st.write("Graph visualization tools go here.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Evolution", "Performance Comparison", "Transaction Volume", "Sector Performance", "Candlestick Chart"])

    # 1. Price Evolution
    with tab1:
        selected_symbol = st.selectbox("Sélectionnez une action", graph_data["Security"].unique(), key="tab1_symbol")
        graph_data_symbol = graph_data[graph_data["Security"] == selected_symbol]
        st.subheader(f"Price Evolution for {selected_symbol}")
        st.line_chart(graph_data_symbol.set_index("Date")["Price Close (USD)"])

    # 2. Performance Comparison
    with tab2:
        st.subheader("Performance Comparison")
        selected_symbols = st.multiselect("Select stocks to compare:", graph_data["Security"].unique(), key="tab2_symbols")
        if selected_symbols:
            graph_data_selected = graph_data[graph_data["Security"].isin(selected_symbols)]
            if not graph_data_selected.empty:
                graph_data_selected["Normalized Close"] = graph_data_selected.groupby("Security")["Price Close (USD)"].transform(lambda x: x / x.iloc[0] * 100)
                fig_perf = px.line(graph_data_selected, x="Date", y="Normalized Close", color="Security", title="Normalized Performance")
                st.plotly_chart(fig_perf)
            else:
                st.warning("No data available for the selected stocks")

    # 3. Transaction Volume
    with tab3:
        st.subheader("Filters")
        selected_symbol = st.selectbox("Select a stock:", graph_data["Security"].unique(), key="shared_symbol")
        selected_year = st.selectbox("Select a year:", sorted(graph_data["Year"].unique()), key="shared_year")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Transaction Volume")
            graph_data_filtered = graph_data[(graph_data["Security"] == selected_symbol) & (graph_data["Year"] == selected_year)]
            if not graph_data_filtered.empty:
                fig = go.Figure(graph_data=[go.Bar(
                    x=graph_data_filtered["Date"],
                    y=graph_data_filtered["Volume"],
                    marker_color='violet'
                )])
                st.plotly_chart(fig)
            else:
                st.warning("No data available for the selected stock and year.")

        with col2:
            st.subheader("High Volume Days")
            graph_data_symbol = graph_data[graph_data["Security"] == selected_symbol]
            high_volume = graph_data_symbol[graph_data_symbol["Year"] == selected_year]
            high_volume = high_volume[high_volume["Volume"] > high_volume["Volume"].quantile(0.95)]
            st.dataframe(high_volume)

    # 4. Sector Performance
    with tab4:
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Sector Performance")
            if "GICS Sector" in graph_data.columns and not graph_data["GICS Sector"].isnull().all():
                graph_data_sector = graph_data.groupby("GICS Sector")["Price Close (USD)"].mean().reset_index()
                fig_bar = px.bar(
                    graph_data_sector,
                    x="GICS Sector",
                    y="Price Close (USD)",
                    title="Average Price by sector",
                    labels={"Price Close (USD)": "Average Price (USD)"},
                    text_auto=True
                )
                fig_bar.update_layout(
                    xaxis=dict(tickangle=45),
                    xaxis_title="",
                    clickmode="event+select"
                )
                fig_bar.update_traces(marker=dict(color="steelblue"), textposition="outside")
                event = plotly_events(fig_bar, override_height=400, key="plotly_events")
                selected_sector = event[0]["x"] if event else None
                st.session_state["selected_sector"] = selected_sector
            else:
                st.warning("Sector data is not available.")

        with col4:
            st.subheader("Sub-Industry Performance")
            if "GICS Sub-Industry" in graph_data.columns and st.session_state.get("selected_sector"):
                selected_sector = st.session_state.get("selected_sector")
                graph_data_filtered = graph_data.groupby(["GICS Sector", "GICS Sub-Industry"])["Price Close (USD)"].mean().reset_index()
                graph_data_filtered = graph_data_filtered[graph_data_filtered["GICS Sector"] == selected_sector]
                fig = px.scatter(
                    graph_data_filtered,
                    x="GICS Sub-Industry",
                    y="Price Close (USD)",
                    size="Price Close (USD)",
                    color="GICS Sub-Industry",
                    title=f"Sub-Industry Performance for {selected_sector} by sub-industry",
                    labels={"Price Close (USD)": "Average Price (USD)", "GICS Sub-Industry": "Sub-industry"},
                    hover_name="GICS Sub-Industry",
                    size_max=50
                )
                fig.update_layout(xaxis=dict(showticklabels=False))
                st.plotly_chart(fig, use_container_width=True)
            elif not st.session_state.get("selected_sector"):
                st.info("Click on a bar in the chart on the left to display the corresponding sub-industries.")
            else:
                st.warning("Sector data is not available.")

    # 5. Candlestick Chart
    with tab5:
        st.subheader("Graphique en chandeliers")

        def get_available_months(year):
            if year == 2019:
                return [11, 12]  # Novembre, Décembre pour 2019
            elif year == 2024:
                return list(range(1, 11))  # Janvier à Octobre pour 2024
            else:
                return list(range(1, 13))  # Tous les mois de 2020 à 2023

        selected_symbol = st.selectbox("Select a stock:", graph_data["Security"].unique(), key="tab5_symbol")
        selected_year = st.selectbox("Select a year:", sorted(graph_data["Year"].unique()), key="tab5_year")
        available_months = get_available_months(selected_year)
        selected_month = st.selectbox("Select a month:", available_months, format_func=lambda m: calendar.month_name[m], key="tab5_month")
        graph_data_filtered = graph_data[
            (graph_data["Security"] == selected_symbol) &
            (graph_data["Year"] == selected_year) &
            (graph_data["Month"] == selected_month)
        ]

        if not graph_data_filtered.empty:
            fig_candle = go.Figure(graph_data=[go.Candlestick(
                x=graph_data_filtered["Date"],
                open=graph_data_filtered["Price Open (USD)"],
                high=graph_data_filtered["Price High (USD)"],
                low=graph_data_filtered["Price Low (USD)"],
                close=graph_data_filtered["Price Close (USD)"]
            )])
            fig_candle.update_layout(
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_candle)
        else:
            st.warning("No data available for the selected month.")


def main():
    st.sidebar.title("S&P500 Visualizer")
    options = ["Main", "Wordcloud", "Equity Screening", "Portfolio", "Graphs"]
    choice = st.sidebar.radio("E", options)

    if choice == "Main":
        st.title("Main")
        st.write("Welcome to the Main page!")
    elif choice == "Wordcloud":
        display_wordcloud()
    elif choice == "Equity Screening":
        display_equity_screening()
    elif choice == "Portfolio":
        display_portfolio()
    elif choice == "Graphs":
        display_graphs()

if __name__ == "__main__":
    main()