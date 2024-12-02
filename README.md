"# Mon Projet"
"# Mon Projet"
# Portfolio Optimization and Analysis Tool

This project provides tools for analyzing historical stock market data, constructing portfolios, and comparing different investment strategies. It is designed for educational and exploratory purposes, showcasing how techniques like Minimum Variance, Equal-Weight, and Equal Risk Contribution (ERC) portfolios can be implemented.

---

## Features

1. **Data Preparation**
   - Reads historical stock data, including price and market capitalization, from an Excel file.
   - Processes and cleans the data by:
     - Converting dates.
     - Handling missing values with forward and backward filling.

2. **Portfolio Construction**
   - Implements three portfolio strategies:
     - **Equal-Weighted Portfolio**
     - **Minimum Variance Portfolio**
     - **Equal Risk Contribution (ERC) Portfolio**

3. **Performance Analysis**
   - Evaluates portfolios based on:
     - Cumulative Returns
     - Annualized Volatility
     - Sharpe Ratio
     - Maximum Drawdown
   - Plots cumulative portfolio values with key metrics annotated.

---

## Installation

1. Clone the repository:
   ```
   git clone <repository_url>
   cd portfolio-optimization
   ```
2. Install required Python libraries:
   ```
   pip install pandas numpy matplotlib scipy
   ```

---

## Usage

### 1. Input Data
Place the stock market data Excel file in the project directory. The default file used is `sbf120_as_of_end_2018.xlsx`.

### 2. Run the Script
Execute the script:
```bash
python portfolio_analysis.py
```

### 3. Output
- **Top 10 Companies**: Displayed in the terminal, ranked by market capitalization as of December 31, 2018.
- **Portfolio Strategies**: Cumulative performance of different strategies plotted, starting at an initial value of \$1,000,000.
- **Performance Metrics**: Sharpe Ratio, Maximum Drawdown, and Annualized Volatility printed for each strategy.

---

## How It Works

### 1. Data Preparation
- Extracts columns for date, price, and market capitalization.
- Combines data from multiple companies into a single DataFrame.
- Filters for the top 10 companies based on their market capitalization.

### 2. Portfolio Strategies
- **Equal-Weighted**: Each stock gets an equal allocation in the portfolio.
- **Minimum Variance**: Portfolio weights are optimized to minimize risk (variance).
- **ERC**: Weights are optimized so that each stock contributes equally to portfolio risk.

### 3. Visualization
- Uses Matplotlib to plot cumulative portfolio values for 2019.
- Annotates key points, such as the maximum drawdown.

---

## Customization

- **Adjust Risk-Free Rate**: Modify `risk_free_rate` to reflect the desired rate.
- **Change Time Period**: Update the date ranges (`top_10_prices_2017_2018` or `top_10_prices_2019`) for analysis.

---

## Example Output

1. **Top Companies:**
   ```
   Top 10 Companies with Highest Market Cap as of 31-Dec-2018:
   Company A: $XXM
   Company B: $XXM
   ```

2. **Portfolio Performance:**
   ```
   Performance of Equal-Weighted Strategy:
   Annualized Volatility: XX.XX%
   Sharpe Ratio: X.XX
   Maximum Drawdown: -XX.XX%
   ```

3. **Plot**:
   - Displays a graph showing the growth of \$1,000,000 for each strategy during 2019.

---

## License

This project is open-source and can be modified or redistributed under the terms of the MIT License.

--- 
