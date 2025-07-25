#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
pd.set_option('display.max_column',None)
pd.set_option('display.width',1000)
import matplotlib.ticker as mtick
from scipy.optimize import minimize
from sklearn.preprocessing import normalize


# In[21]:


# Load and preprocess data
df = pd.read_csv('C:/Users/prakh/Downloads/data.csv', parse_dates=['Date '], index_col='Date ')
df = df.sort_index()
df = df.dropna()
display(df)


# In[22]:


# Calculate log returns
returns = np.log(df / df.shift(1)).dropna()
returns.describe()


# In[23]:


# Check for missing values 
print(df.isnull().sum())


# In[24]:


# Use sklearn PCA for variance explained
cov_matrix = returns.cov()
pca = PCA()
pca.fit(returns)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


# In[25]:


# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
variance_table = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance Ratio': np.round(explained_variance, 4),
    'Cumulative Variance': np.round(cumulative_variance, 4)
})
print(variance_table.head(10))
plt.plot(cumulative_variance, marker='o')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Variance Explained')
plt.show()


# In[26]:


# Correlation and covariance matrices
correlation_matrix = returns.corr()
plt.figure(figsize=(14, 10))  # Wider figure
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Daily Log Returns", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Convert to form
corr_pairs = correlation_matrix.unstack().reset_index()
corr_pairs.columns = ['Stock 1', 'Stock 2', 'Correlation']

# Remove  duplicates
corr_pairs = corr_pairs[corr_pairs['Stock 1'] != corr_pairs['Stock 2']]
corr_pairs = corr_pairs.drop_duplicates(subset=['Correlation'])

# Top 10 positive and top 10 negative correlations
top_positive = corr_pairs.sort_values(by='Correlation', ascending=False).head(10).copy()
top_positive['Type'] = 'Positive'

top_negative = corr_pairs.sort_values(by='Correlation', ascending=True).head(10).copy()
top_negative['Type'] = 'Negative'

# Combine
highlighted_corrs = pd.concat([top_positive, top_negative], ignore_index=True)
highlighted_corrs = highlighted_corrs[['Type', 'Stock 1', 'Stock 2', 'Correlation']]

# Show result
print("\n Top Positively and Negatively Correlated Stock Pairs:\n")
print(highlighted_corrs)


# In[27]:


# Reduce dimensions
k = 3
returns_reduced = pca.transform(returns)[:, :k]
eigenportfolios = eigenvectors.T


# In[28]:


# Calculate variance explained by eigenportfolios
normalized_portfolios = np.array([normalize(w.reshape(1, -1))[0] for w in eigenportfolios])

portfolio_returns = returns @ normalized_portfolios.T
portfolio_variances = np.var(portfolio_returns, axis=0)
variance_explained = portfolio_variances / np.sum(portfolio_variances)

# Cumulative variance
cumulative_variance = np.cumsum(variance_explained)

# Create table
eigen_table = pd.DataFrame({
    'Eigenportfolio #': [f'EP{i+1}' for i in range(len(variance_explained))],
    'Variance Explained (%)': np.round(variance_explained * 100, 2),
    'Cumulative Variance (%)': np.round(cumulative_variance * 100, 2)
})

# Display table
print("\n  Variance Explained by Each Eigenportfolio:\n")
print(eigen_table)

# Plot
plt.figure(figsize=(10, 5))
bars = plt.bar(eigen_table['Eigenportfolio #'], variance_explained, color='skyblue')
plt.title('Variance Explained by Eigenportfolios')
plt.ylabel('Proportion of Variance')
plt.xticks(rotation=45)

# % labels on top of bars
for bar, value in zip(bars, variance_explained):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{value:.2%}', 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# In[29]:


# the inverse of the covariance matrix
inv_cov = np.linalg.inv(cov_matrix)

# a vector of ones (for equal weights constraint)
ones = np.ones(len(cov_matrix))

#  Compute MVP weights using the formula:
# w_mvp = (Σ⁻¹1) / (1ᵗΣ⁻¹1)
w_mvp = inv_cov @ ones
w_mvp /= ones.T @ inv_cov @ ones

#  a DataFrame to display weights nicely
mvp_weights = pd.DataFrame({
    'Stock': df.columns,
    'MVP Weight (%)': np.round(w_mvp * 100, 4)
}).sort_values(by='MVP Weight (%)', ascending=False)

# Display the weights
print("\n Minimum Variance Portfolio (MVP) Weights:\n")
print(mvp_weights)


# Plot MVP weights
plt.figure(figsize=(10, 5))
sns.barplot(data=mvp_weights, x='Stock', y='MVP Weight (%)', palette='viridis')
plt.title("Minimum Variance Portfolio Weights")
plt.ylabel("Weight (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


# equal weights
n_assets = len(df.columns)
equal_weights = np.ones(n_assets) / n_assets

#  a DataFrame of weights for clarity
equal_weight_df = pd.DataFrame({
    'Stock': df.columns,
    'Equal Weight (%)': np.round(equal_weights * 100, 2)
})

# Display the weights
print("\n📊 Equal Weight Portfolio Weights:\n")
print(equal_weight_df)

#  Plot Equal Weights
plt.figure(figsize=(10, 5))
sns.barplot(data=equal_weight_df, x='Stock', y='Equal Weight (%)', palette='crest')
plt.title("Equal Weight Portfolio Allocation")
plt.ylabel("Weight (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[31]:


#  Minimum Variance Portfolio
portfolio_mvp = returns @ w_mvp

#  Top Eigenportfolio (Highest eigenvalue = first component, not last)
top_eigenportfolio_vector = eigenvectors[:, 0]  
top_eigenportfolio = returns @ top_eigenportfolio_vector

# Equal Weight Portfolio
portfolio_equal = returns @ equal_weights

# DataFrame of portfolio returns
portfolio_returns_df = pd.DataFrame({
    'Date': returns.index,
    'MVP': portfolio_mvp,
    'Top Eigenportfolio': top_eigenportfolio,
    'Equal Weight': portfolio_equal
}).set_index('Date')

# Print summary statistics
print("\n  Portfolio Return Statistics:\n")
print(portfolio_returns_df.describe())

#  Cumulative Returns Plot
portfolio_cumulative = (1 + portfolio_returns_df).cumprod() - 1

plt.figure(figsize=(10, 5))
for col in portfolio_cumulative.columns:
    plt.plot(portfolio_cumulative.index, portfolio_cumulative[col], label=col)

plt.title(" Cumulative Returns of Portfolios")
plt.ylabel("Cumulative Return")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[32]:


# Assume annual risk-free rate of 3.5%
annual_rf = 0.035

#  Convert to daily (252 trading days in a year)
risk_free_rate_daily = annual_rf / 252

print(f"Daily Risk-Free Rate: {risk_free_rate_daily:.6f}")


# In[36]:


# Sharpe Ratio function with both daily /  annualized output
def sharpe_ratio(returns, rf=risk_free_rate_daily, annualize=True):
    excess_returns = returns - rf
    mean_daily = excess_returns.mean()
    std_daily = excess_returns.std()
    
    daily_sharpe = mean_daily / std_daily
    
    if annualize:
        annual_sharpe = daily_sharpe * np.sqrt(252)
        return round(mean_daily, 6), round(std_daily, 6), round(daily_sharpe, 4), round(annual_sharpe, 4)
    else:
        return round(mean_daily, 6), round(std_daily, 6), round(daily_sharpe, 4), None



# Dictionary of portfolios
portfolios = {
    'MVP': portfolio_mvp,
    'Top Eigenportfolio': top_eigenportfolio,
    'Equal Weight': portfolio_equal
}

# Store results
sharpe_results = []

for name, ret in portfolios.items():
    mean, std, daily_sharpe, annual_sharpe = sharpe_ratio(ret)
    sharpe_results.append({
        'Portfolio': name,
        'Mean Daily Return': mean,
        'Standard Deviation': std,
        'Daily Sharpe': daily_sharpe,
        'Annualized Sharpe': annual_sharpe
    })

# Create DataFrame
sharpe_df = pd.DataFrame(sharpe_results)
# Reorder columns and format nicely
sharpe_df = sharpe_df[
    ['Portfolio', 'Mean Daily Return', 'Standard Deviation', 'Daily Sharpe', 'Annualized Sharpe']
]
# Format numbers to 4 decimal places
sharpe_df['Mean Daily Return'] = sharpe_df['Mean Daily Return'].map('{:.6f}'.format)
sharpe_df['Standard Deviation'] = sharpe_df['Standard Deviation'].map('{:.6f}'.format)
sharpe_df['Daily Sharpe'] = sharpe_df['Daily Sharpe'].map('{:.4f}'.format)
sharpe_df['Annualized Sharpe'] = sharpe_df['Annualized Sharpe'].map('{:.4f}'.format)

# Display table
print("\n Portfolio Risk & Return Metrics:\n")
display(sharpe_df)





# In[40]:


# Recreate DataFrame without formatting
sharpe_df = pd.DataFrame(sharpe_results)

# Plot without converting to string
sns.set(style="whitegrid")

# Sort by Sharpe Ratio (numeric)
sharpe_df_sorted = sharpe_df.sort_values(by='Daily Sharpe', ascending=False)

# Plot
plt.figure(figsize=(9, 5))
barplot = sns.barplot(x='Portfolio', y='Daily Sharpe', data=sharpe_df_sorted, palette='Blues_d')

# Add value labels (formatted nicely, but values remain numeric)
for i, value in enumerate(sharpe_df_sorted['Daily Sharpe']):
    plt.text(i, value + 0.002, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

# Labels and styling
plt.title("📈 Daily Sharpe Ratio Comparison of Portfolios", fontsize=14)
plt.xlabel("Portfolio Strategy", fontsize=12)
plt.ylabel("Sharpe Ratio (Daily)", fontsize=12)
plt.ylim(0, sharpe_df_sorted['Daily Sharpe'].max() + 0.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
# For table display only
display(sharpe_df.style.format({
    'Mean Daily Return': "{:.6f}",
    'Standard Deviation': "{:.6f}",
    'Daily Sharpe': "{:.4f}",
    'Annualized Sharpe': "{:.4f}"
}))




# In[41]:


# Objective: Negative Sharpe (to maximize it via minimize)
def neg_sharpe_ratio(weights, mu, cov, rf):
    port_return = np.dot(weights, mu)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return -(port_return - rf) / port_volatility
mu = returns.mean() * 252
cov = returns.cov() * 252
n_assets = len(mu)
initial_weights = np.ones(n_assets) / n_assets
bounds = tuple((0, 1) for _ in range(n_assets))
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

result = minimize(
    neg_sharpe_ratio,
    initial_weights,
    args=(mu, cov, annual_rf),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

markowitz_weights = result.x


# In[42]:


portfolio_markowitz = returns @ markowitz_weights


# In[43]:


portfolio_returns_df['Markowitz'] = portfolio_markowitz


# In[44]:


portfolio_cumulative = (1 + portfolio_returns_df).cumprod() - 1


# In[49]:


portfolios = {
    'MVP': portfolio_mvp,
    'Top Eigenportfolio': top_eigenportfolio,
    'Equal Weight': portfolio_equal,
    'Markowitz': portfolio_markowitz
}



# In[52]:


# Sort weights and format
markowitz_df = pd.DataFrame({
    'Stock': df.columns,
    'Markowitz Weight (%)': np.round(markowitz_weights * 100, 2)  # Assuming `markowitz_weights` is your numpy array
})

# Filter non-zero weights only
markowitz_df = markowitz_df[markowitz_df['Markowitz Weight (%)'] > 0]

# Sort by weight descending
markowitz_df = markowitz_df.sort_values(by='Markowitz Weight (%)', ascending=False).reset_index(drop=True)

# Display neatly
display(markowitz_df.style.set_caption("Markowitz Portfolio Weights").format({
    'Markowitz Weight (%)': "{:.2f}"
}))
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Markowitz Weight (%)', 
    y='Stock', 
    data=markowitz_df, 
    palette='viridis'
)

# Add weight labels
for i, row in markowitz_df.iterrows():
    plt.text(row['Markowitz Weight (%)'] + 0.5, i, f"{row['Markowitz Weight (%)']:.2f}%", va='center')

plt.title("Markowitz Portfolio Allocation", fontsize=14)
plt.xlabel("Weight (%)", fontsize=12)
plt.ylabel("Stock", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# In[53]:


# Combine weights into a single DataFrame
comparison_df = pd.DataFrame({
    'Stock': df.columns,
    'Markowitz': markowitz_weights * 100,
    'MVP': w_mvp * 100,
    'Equal Weight': equal_weights * 100
})

# Reshape to long-form for seaborn
comparison_long = comparison_df.melt(id_vars='Stock', var_name='Portfolio Type', value_name='Weight (%)')

# Filter out 0% weights (optional, for cleaner plot)
comparison_long = comparison_long[comparison_long['Weight (%)'] > 0]


# In[54]:


plt.figure(figsize=(12, 6))
sns.barplot(
    data=comparison_long,
    x='Stock',
    y='Weight (%)',
    hue='Portfolio Type',
    palette='Set2'
)

plt.title("Comparison of Portfolio Weights: Markowitz vs MVP vs Equal Weight", fontsize=14)
plt.xlabel("Stock", fontsize=12)
plt.ylabel("Portfolio Allocation (%)", fontsize=12)
plt.xticks(rotation=45)
plt.legend(title="Portfolio Type")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# In[61]:


# --------------------------------------------------------
# Cumulative Return Comparison: Markowitz vs MVP vs Equal Weight
# --------------------------------------------------------

# STEP 1: Calculate cumulative returns (if not done already)
# This assumes daily returns for each portfolio are already defined as:
# portfolio_markowitz, portfolio_mvp, portfolio_equal

cumulative_returns = pd.DataFrame({
    'Markowitz': (1 + portfolio_markowitz).cumprod(),
    'MVP': (1 + portfolio_mvp).cumprod(),
    'Equal Weight': (1 + portfolio_equal).cumprod()
})

# STEP 2: Plot cumulative returns for visual comparison
plt.figure(figsize=(12, 6))

# Color coding for each portfolio
colors = {
    'Markowitz': 'royalblue',
    'MVP': 'orangered',
    'Equal Weight': 'forestgreen'
}

# Line plot for each strategy
for col in cumulative_returns.columns:
    plt.plot(
        cumulative_returns.index,
        cumulative_returns[col],
        label=col,
        color=colors[col],
        linewidth=2.5
    )
    # Add final value as label at the end
    plt.text(
        cumulative_returns.index[-1],
        cumulative_returns[col].iloc[-1],
        f'{cumulative_returns[col].iloc[-1]:.2f}x',
        fontsize=10,
        color=colors[col],
        va='center',
        ha='left'
    )

# Axis labels and title
plt.title("Cumulative Return Comparison: Markowitz vs MVP vs Equal Weight", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Portfolio Value (₹1 Initial Investment)", fontsize=12)

# Legend and grid
plt.legend(title="Portfolio Strategy", loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------
#This graph visualizes the performance of three portfolio strategies:
#1. Markowitz Mean-Variance Optimal Portfolio
#2. Minimum Variance Portfolio (MVP)
#3. Equal Weight Portfolio

#-------------------------------------------------------------------
#Each line tracks how ₹1 invested at the beginning grows over time under each strategy. 
#This helps in comparing overall performance visually and evaluating consistency, volatility, 
#and total return of each method.
#-----------------------------------------------------------------------
#Key Insight:
#- A higher ending value (e.g., #1.55x) means that ₹1 grew to ₹1.55.
#- Markowitz often targets maximum Sharpe ratio, MVP minimizes risk, 
 # and Equal Weight ensures diversification without optimization.
#----------------------------------------------------------------------

# --------------------------------------------------------
# Create a Comparison Table of Final Portfolio Values
# --------------------------------------------------------

final_values = cumulative_returns.iloc[-1].reset_index()
final_values.columns = ['Portfolio', 'Final Value (₹)']
final_values['Return (%)'] = ((final_values['Final Value (₹)'] - 1) * 100).round(2)

print("\n Final Portfolio Comparison:\n")
display(final_values.sort_values(by='Return (%)', ascending=False))




# In[ ]:




