# Introduction

Options are financial contracts that grant the holder the right, but not the obligation, to purchase or sell an underlying security at a predetermined price and date. Many models have been developed to valuate these contracts, one of them being the Black-Scholes framework. In addition to pricing these contracts, the Black-Scholes model also gives us a way to gain insight into the risks an option faces. These quantities are popularly called the "Greeks" and they are measures of the changes in the options values to changes in any of its parameters and can be utilized to neatralize certain risks. 

BS_options info.py includes a simple function that calculates the most popular greeks of a call and a put option along with their respective price and standard deviation. The function also takes a boolean input to plot the zeroth, first and second derivative of the option price with respect to the underlying.

A second script simulates the overnight holding profit of an unhegded written call along with two different hedging strategies.
