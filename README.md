# Dynamic Ticket Pricing using Q-Learning

This project implements a Q-learning-based dynamic pricing model for optimizing ticket prices in a simulated environment. It considers factors such as customer segments, competitor pricing, booking rates, and time horizons to adjust ticket prices and maximize revenue.

## Features
- **Dynamic Pricing**: Incorporates non-linear price sensitivity and competitor pricing dynamics.
- **Customer Segmentation**: Differentiates between Economy and Business customers.
- **Q-Learning Algorithm**: Trains an optimal pricing policy through reinforcement learning.
- **Stochastic Components**: Adds randomness to simulate real-world uncertainties in demand and competitor actions.

---

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Core Components](#core-components)
- [Q-Learning Parameters](#q-learning-parameters)
- [Optimal Pricing Policy](#optimal-pricing-policy)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Requirements

The following Python libraries are required to run this project:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `pulp`
- `random`

Install the dependencies using:
```bash
pip install numpy pandas matplotlib seaborn pulp

## Usage

### Set Parameters:
- **Adjust the parameters** like a seasoned chef tweaking a recipe:
  - `num_seats`: The total number of tickets available. Think of this as your inventory of dreams.
  - `time_horizon`: The number of time periods for ticket sales. Do you want tickets to fly off the shelves today, or trickle out over time?
  - `price_levels`: Set the potential ticket price points. Yes, you are the puppeteer pulling the strings of revenue.

### Run the Model:
1. The Q-learning algorithm will simulate ticket sales over multiple episodes. Think of it as the "Rocky Balboa" of algorithms—training tirelessly to master the fight.
2. At the end of this rigorous training, an optimal pricing policy emerges like a phoenix, ready to maximize your revenue.

### Output:
- Behold the wisdom of the Q-learning oracle! It will deliver the optimal pricing policy for any given state, based on factors like:
  - Seats left
  - Booking rates
  - Competitor pricing
  - Customer segments
  - And other juicy metrics.

To start your dynamic pricing journey, execute:
```bash
python code.py

## Core Components

### 1. Demand Probability Calculation
The function `get_demand_probability` estimates the likelihood of ticket purchase using:
- **Price Sensitivity**: Higher sensitivity for Economy customers, reflecting tighter budget constraints.
- **Competitor Influence**: Adjusts demand based on the pricing levels of competitors.
- **Customer Elasticity**: Models non-linear scaling of demand depending on the customer type.

### 2. Booking Rate Estimation
The function `get_booking_rate` estimates how the booking rate changes over time, based on:
- **Seats Sold**: As tickets sell out, demand tends to increase due to scarcity.
- **Time Elapsed**: Time has a logarithmic influence, with customers growing more eager as the travel date approaches.

### 3. Competitor Pricing Dynamics
The function `get_competitor_price_level` simulates how competitor prices evolve:
- **Time Progression**: Gradual pricing adjustments throughout the time horizon.
- **Random Fluctuations**: Adds randomness to reflect the uncertainty in competitor strategies.

### 4. Customer Segmentation
The function `get_customer_segment` classifies customers into Economy or Business categories, based on:
- **Time**: Business travelers are more likely to book closer to the travel date.
- **Seats Left**: A higher likelihood of Business customers as ticket availability decreases.

---

## Q-Learning Parameters

### Learning Configuration
The algorithm trains under the following parameters:
- **Learning Rate**: `0.15` – Controls how much new information overrides old information.
- **Discount Factor**: `0.85` – Determines the importance of future rewards versus immediate rewards.
- **Exploration Rate**: `0.2` – Balances exploration (trying new actions) and exploitation (choosing the best-known actions).
- **Episodes**: `5000` – Number of training iterations to optimize the pricing policy.

### States and Actions
- **States**: Represented as combinations of:
  - Seats left
  - Time remaining
  - Booking rates
  - Competitor pricing
  - Customer segments
- **Actions**: Correspond to predefined ticket price levels.

### Reward Mechanism
- **Successful Sale**: Positive reward equal to the ticket price.
- **Unsuccessful Sale**: No reward is provided.

---

## Optimal Pricing Policy

The trained Q-learning algorithm outputs an optimal pricing policy by suggesting the price level that maximizes revenue for a given state.

### Example Scenario
- **Seats left**: 50  
- **Time remaining**: 5 days  
- **Booking rate**: Medium  
- **Competitor pricing**: Low  
- **Customer segment**: Economy  

**Outcome**: The algorithm identifies the best price level based on the learned Q-values, ensuring maximum profitability.

---

## Future Improvements

Here are potential areas for enhancement:
1. **Expand Customer Segments**: Introduce First-Class or other sub-segments to better reflect real-world diversity.
2. **Increase Time Granularity**: Divide time into finer intervals for more precise pricing adjustments.
3. **Dynamic Competitor Behavior**: Simulate more advanced and realistic competitor strategies to challenge the model.
4. **Incorporate Real-World Data**: Use historical sales data to enhance the accuracy and robustness of predictions.

