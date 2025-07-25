# PCA-Portfolio-Optimization

# Portfolio Strategies Compared

| Portfolio Type           | Strategy                                              |
|--------------------------|-------------------------------------------------------|
| Equal Weight             | 1/n allocation across all assets                      |
| Minimum Variance (MVP)   | Portfolio that minimizes total risk (σ²)             |
| PCA-Based Eigenportfolio | Top principal component portfolio                     |
| Markowitz Optimal        | Efficient frontier maximizing Sharpe Ratio           |



##  Visualizations

- 🔷 **Cumulative Return Graph**: To compare long-term growth
- 🔷 **Sharpe Ratio Bar Plot**: To evaluate risk-adjusted performance
- 🔷 **Variance Explained Plot**: Shows dominance of first few PCs
- 🔷 **Portfolio Weight Allocations**: Visualizing asset importance

---

##  Results Summary

| Portfolio            | Mean Daily Return | Annualized Sharpe Ratio |
|----------------------|------------------:|-------------------------:|
| Markowitz Efficient  | 0.00082           | 1.42                     |
| Minimum Variance     | 0.00064           | 1.28                     |
| Equal Weight         | 0.00060           | 1.12                     |
| Top Eigenportfolio   | 0.00078           | 1.36                     |

> **Conclusion**: PCA-based portfolios showed competitive performance with lower dimensionality. Markowitz still performed best on Sharpe Ratio but with higher volatility.

---

##  Future Work

- Implement Value at Risk (VaR) and Conditional VaR (CVaR)
- Use PCA on fundamental ratios (like PE, PB, etc.)
- Apply LSTM models for future return forecasting
- Compare with real-world ETF benchmarks



# About the Author

**Prakhar Shukla**  
BSc Mathematics (Hons) + Minor in Economics [ University Of Delhi ] 
Researcher 
📍 delhi, India  
🔗 [LinkedIn](www.linkedin.com/in/prakharshukla1354)

---

## 📜 License

This project is intended for academic and educational purposes.  
© 2025 [Prakhar Shukla]. All rights reserved.
