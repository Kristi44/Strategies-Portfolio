import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.ticker as ticker

df = pd.read_excel("C:/Users/crist/Downloads/sbf120_as_of_end_2018.xlsx", sheet_name=None)

#print(df.keys())
compagnies=df['Compo 31122018'].copy()
dataset=df['Data'].copy()
#print(dataset)
#print(dataset.dtypes)

date_colums =[3*i for i in range(120)]
for i in date_colums:
    dataset.iloc[:, i] = pd.to_datetime(dataset.iloc[:, i].astype(float), origin='1899-12-30', unit='D')
#print(dataset)
#print(dataset.dtypes)


dfs=[]
#Loop through the 120 sets of 3 colums (each set has adate, price, and an extra column)

for i in range(0,360,3):
    "Extract the relevant date and price colums, and market capitalisation"
    date_col=dataset.iloc[:, i]#Date colum
    price_col=dataset.iloc[:, i+1]#price column
    mark_cap_col=dataset.iloc[:, i+2]#market capitalization column
    "Create a temporary DataFrame with these three columns"
    temp_df=pd.DataFrame({
        'Date': pd.to_datetime(date_col, errors='coerce'),
    f'Price {dataset.columns[i]}': price_col,
    f'Market Cap {dataset.columns[i]}': mark_cap_col
    })

    temp_df.dropna(inplace=True)
    temp_df.set_index('Date', inplace=True)

    'Append the DataFrame to the list'
    dfs.append(temp_df)

result = dfs[0]
for i in range(1,120):
    result = result.join(dfs[i], how='outer')

for col in result.columns:
    #Find the first and last non-Nan index for each column
    first_valid_idx=result[col].first_valid_index()
    last_valid_idx=result[col].last_valid_index()

    #If there are valid indices (i.e, the colum is not entirely Nan)
    if first_valid_idx is not None and last_valid_idx is not None:
        #Apply forward fill only betwenn the first and last valid indices
        result.loc[first_valid_idx:last_valid_idx,col]= result.loc[first_valid_idx:last_valid_idx,col].bfill()

#print(result)
#print(result.columns)

# Filter for Market Cap columns
market_cap_cols = [col for col in result.columns if "Market Cap" in col]

# Extract market caps as of December 31, 2018
end_2018_market_caps = result.loc['2018-12-31', market_cap_cols]

# Find the 10 companies with the highest market capitalization
top_10_companies = end_2018_market_caps.nlargest(10)

# Display the top 10 companies and their market caps
print("Top 10 Companies with Highest Market Cap as of 31-Dec-2018:")
print(top_10_companies)

# Retrieve the corresponding price columns
top_10_price_cols = [col.replace("Market Cap", "Price") for col in top_10_companies.index]

# Filter the result DataFrame for only the top 10 companies' price data
top_10_prices = result[top_10_price_cols]
top_10_prices_2019 = top_10_prices.loc['2019-01-01':'2019-12-31']
top_10_prices_2017_2018 = top_10_prices.loc['2017-01-01':'2018-12-31']
daily_returns_2017_2018 = top_10_prices_2017_2018.pct_change().dropna()
daily_returns_2019 = top_10_prices_2019.pct_change().dropna()


V0 = 1_000_000  # Valeur initiale du portefeuille
risk_free_rate = 0.03  # Taux sans risque (exemple de 1% annuel)

class Strategy:
    def __init__(self, name, returns, risk_free_rate=0.03):
        self.name = name
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.cumulative_values = V0 * (1 + self.returns).cumprod()

    def annual_volatility(self):
        return self.returns.std() * np.sqrt(255)

    def sharpe_ratio(self):
        annual_return = ((self.cumulative_values.iloc[-1] - V0)/V0)  # Annualized for single year
        return (annual_return - self.risk_free_rate) / self.annual_volatility()

    def max_drawdown(self):
        running_max = self.cumulative_values.cummax()
        drawdown = (self.cumulative_values / running_max) - 1
        max_drawdown_value = drawdown.min()
        max_drawdown_date = drawdown.idxmin()  # Get the date of max drawdown
        return max_drawdown_value, max_drawdown_date

    def plot(self, ax):
        """
        Plot the cumulative portfolio values on a given axis.
        :param ax: Matplotlib axis to plot on.
        """
        # Plot cumulative returns curve
        ax.plot(
            self.cumulative_values,
            label=f"{self.name} (Sharpe: {self.sharpe_ratio():.2f}, Max Drawdown: {self.max_drawdown()[0]:.2%})"
        )

        # Get the max drawdown point
        max_drawdown_value, max_drawdown_point = self.max_drawdown()

        # Plot the max drawdown point
        ax.plot(max_drawdown_point, self.cumulative_values[max_drawdown_point], 'ro')  # Red dot for max drawdown
        ax.text(max_drawdown_point, self.cumulative_values[max_drawdown_point],
                f"Max DD: {max_drawdown_value:.2%}",
                horizontalalignment='right', fontsize=10, color='red')

        # Format y-axis in millions
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))


# Fonction pour le portefeuille de variance minimale
def minimum_variance(ret):
    def find_port_variance(weights):
        cov = ret.cov()
        port_var = np.dot(weights.T, np.dot(cov, weights)) * 255
        return port_var

    def weight_cons(weights):
        return np.sum(weights) - 1

    bounds_lim = [(0, 1) for _ in range(len(ret.columns))]
    init = [1 / len(ret.columns) for _ in range(len(ret.columns))]
    constraint = {'type': 'eq', 'fun': weight_cons}

    optimal = minimize(fun=find_port_variance,
                       x0=init,
                       bounds=bounds_lim,
                       constraints=constraint,
                       method='SLSQP')

    return list(optimal['x'])

# Fonction pour calculer les poids ERC
def calc_weights_erc(cov):
    def fun(x):
        risk_contributions = x.dot(cov) * x
        risk_diffs = np.reshape(risk_contributions, (len(risk_contributions), 1)) - \
                     np.reshape(risk_contributions, (1, len(risk_contributions)))
        return np.sum(np.square(risk_diffs)) / 10000

    N = cov.shape[0]
    x0 = 1 / np.sqrt(np.diag(cov))
    x0 = x0 / x0.sum()

    bounds = [(0, 1) for _ in range(N)]
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    return res.x



# Rendements journaliers pour chaque stratégie

# 1. Stratégie Équi-Pondérée
equal_weights = np.repeat(1 / 10, 10)
portfolio_daily_returns_equi = daily_returns_2019.dot(equal_weights)

# 2. Stratégie Variance Minimale
weights_min_var = minimum_variance(daily_returns_2017_2018)
portfolio_daily_returns_min_var = daily_returns_2019.dot(weights_min_var)

# 3. Stratégie ERC
cov_matrix_2017_2018 = daily_returns_2017_2018.cov()
weights_erc = calc_weights_erc(cov_matrix_2017_2018)
portfolio_daily_returns_erc = daily_returns_2019.dot(weights_erc)

# Création des stratégies
strategy_equi = Strategy("Équi-Pondéré", portfolio_daily_returns_equi)
strategy_min_var = Strategy("Variance Minimale", portfolio_daily_returns_min_var)
strategy_erc = Strategy("ERC", portfolio_daily_returns_erc)

# Graphique des trois stratégies
fig, ax = plt.subplots(figsize=(12, 8))

for strategy in [strategy_equi, strategy_min_var, strategy_erc]:
    strategy.plot(ax)

ax.set_title("Performance Cumulative des Portefeuilles (2019)")
ax.set_xlabel("Date")
ax.set_ylabel("Valeur Cumulative")
ax.legend()
plt.show()

# Résultats des stratégies
for strategy in [strategy_equi, strategy_min_var, strategy_erc]:
    print(f"Performance of {strategy.name} Strategy:")
    print(f"Annualized Volatility: {strategy.annual_volatility():.2%}")
    print(f"Sharpe Ratio: {strategy.sharpe_ratio():.2f}")
    print(f"Maximum Drawdown: {strategy.max_drawdown()[0]:.2%}")
    print("-" * 50)

print(f" The weights of the Minimum Variance Portfolio are  : {weights_min_var}")
print(f"The weights of the ERC Portfolio are: {weights_erc}")

print(strategy_erc.cumulative_values.iloc[-1])
print(strategy_erc.cumulative_values.iloc[1])

