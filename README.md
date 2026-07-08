Contains the code for the assignments done as a part of the FT5010 class.

Final group project

Defensive Technical Analysis Trading Strategy

A Python-based quantitative trading system that implements a defensive technical analysis strategy for a portfolio of six large-cap Indian information technology companies listed on the National Stock Exchange (NSE). The project combines multiple technical indicators to generate trading signals, incorporates practical risk management mechanisms, and evaluates its effectiveness through historical backtesting and benchmark comparisons. The strategy was developed as part of the FE5221 – Trading Principles & Fundamentals course at the National University of Singapore.

Project Overview

The objective of this project is to design and evaluate a rule-based trading strategy capable of identifying profitable trading opportunities while limiting downside risk. Rather than relying on a single technical indicator, the strategy combines three widely used momentum and trend indicators:

Relative Strength Index (RSI) to identify overbought and oversold market conditions.
Moving Average Convergence Divergence (MACD) to detect trend changes and momentum shifts.
Stochastic Oscillator to evaluate price momentum relative to recent trading ranges.

Trades are executed only when at least two of the three indicators agree on a buy or sell signal, reducing false positives and improving decision quality. Additionally, the strategy incorporates a market crash detection mechanism that temporarily suspends trading during periods of extreme market declines, providing an additional layer of risk protection.

Portfolio

The strategy is evaluated on an equally weighted portfolio consisting of six of India's largest publicly listed IT companies:

Tata Consultancy Services (TCS)
Infosys
HCL Technologies
Wipro
LTIMindtree
Tech Mahindra

These companies were selected due to their high liquidity, significant market capitalisation, and long historical price records, making them well suited for quantitative strategy development and backtesting.

Strategy Evaluation

The trading strategy was initially backtested using historical market data from 2023 to optimise indicator parameters and validate trading rules. It was subsequently evaluated on unseen market data from 19 August 2024 to 15 November 2024 using an initial portfolio value of INR 1,000,000. Performance was compared against two benchmark approaches:

A passive buy-and-hold portfolio with equal allocation across the six stocks.
An equivalent investment in the Nifty50 index.

The implementation also models realistic trading conditions by accounting for transaction costs and assessing performance using standard financial metrics such as returns, Sharpe Ratio, Sortino Ratio, Alpha, Beta, and portfolio volatility.

Key Highlights
Multi-indicator technical trading strategy
Consensus-based buy and sell signal generation
Portfolio-level backtesting framework
Crash detection and trading suspension mechanism
Transaction cost modelling
Benchmark comparison against passive investing and the Nifty50 index
Performance evaluation using common quantitative finance metrics
