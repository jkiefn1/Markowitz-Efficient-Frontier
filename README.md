# Markowitz-Efficient-Frontier
A python program to output the 'efficient frontier' of investment choices. Inspired by Markowitz's work on Modern Portfolio Theory.

# Text-Similarity
## Table of contents
* [Disclaimer](#disclaimer)
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## Disclaimer
I am not a financial professional, and have never claimed to be. This prgoram is meant to be illustrative only, and I **DO NOT** guarantee that this code is flawless or accurate, especially as it pertains to investment guidance. Please use this code at your own discretion, as I will not guarantee any accuracy or precision.

To be able to run the code, you must acknowledge the above by entering passing 'yes' to the acknowledgement variable.

## General info
This program is designed to explore Markowitz's Modern Portfolio Theory on the effifient frontier of investments-that there is an abolustely optimal mix of funds for a desired amount of risk, and that all other mixtures of said funds are un-optimal. This program displays:
* The correlation between passed funds
* The growth of each fund over the specified timeframe
* The efficient frontier of investment allocations for random mixtures of each fund in a portfolio
* The growth of the top simulated portfolios, by return and by risk-weighted return.

## Technologies
Project is created with:
* Pandas 1.2.0
* Numpy 1.19.5
* yfinance 0.1.54
* Plotly 4.14.3
	
## Setup
To run this project, clone and run via your preferred Python environment. The below steps may assist you:

```
https://github.com/jkiefn1/Markowitz-Efficient-Frontier.git
cd Markowitz-Efficient-Frontier
$ python3 Efficient-Frontier.py
```
