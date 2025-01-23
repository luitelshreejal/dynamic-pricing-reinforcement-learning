import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, LpStatus

num_seats = 100 # Initialize the total number of seats available (Can change if want)
time_horizon = 12 # Time horizon (represents a single day of ticket sales) (Big assumption)
price_levels = list(range(100, 650+1, 20)) # Available price levels for tickets (ceiling levels by $20) (arbitrary choice)
customer_segments = ['Economy', 'Business'] # Should we add first-class?
booking_rates = ['Low', 'Medium', 'High'] # Dependent on # seats_sold and time elasped in the horizon
competitor_prices = ['Low', 'Medium', 'High'] # Dependent on time 

def get_demand_probability(price, customer_segment, competitor_price_level):
    """
    Calculates demand probability incorporating nonlinear price sensitivity and competitor dynamics.
    - `base_prob`: Represents the baseline demand probability (50%).
    - `price_sensitivity`: Determines how price changes affect the demand probability, varying by customer segment:
      - Higher sensitivity for 'Economy' (0.002) than 'Business' (0.001).
    - `competitor_influence`: Adjusts demand based on competitor pricing:
      - Lower prices (e.g., 'Low') reduce demand by 25% (multiplier of 0.75).
      - Higher prices (e.g., 'High') increase demand by 30% (multiplier of 1.3).
    - `customer_elasticity`: Applies nonlinear scaling of price sensitivity:
      - Greater elasticity for 'Economy' (1.2) than 'Business' (0.8).
    - Returns a probability between 0.1 and 1.0 after applying all factors.
    """
    base_prob = 0.5
    price_sensitivity = {'Economy': 0.002, 'Business': 0.001}[customer_segment]
    competitor_influence = {'Low': 0.75, 'Medium': 1.05, 'High': 1.3}[competitor_price_level]
    customer_elasticity = 1.2 if customer_segment == 'Economy' else 0.8
    demand_prob = base_prob * (np.exp(-price_sensitivity * price ** customer_elasticity) + 0.1) * competitor_influence
    return min(max(demand_prob, 0.1), 1.0)

def get_booking_rate(seats_sold, time_elapsed):
    """
    Categorizes booking rates using non-linear thresholds based on time elapsed and dynamic seat adjustment.
    - `time_factor`: Calculates the logarithmic effect of elapsed time on the booking rate.
      - Prevents extremely high rates at low time intervals.
    - `rate`: The booking rate, derived from the number of seats sold divided by time adjusted by `time_factor`.
    - Thresholds:
      - 'Low' if the rate is below 25% of expected average sales (num_seats / (4 * time_horizon)).
      - 'Medium' for rates below 60% of expected average sales ((3 * num_seats) / (5 * time_horizon)).
      - 'High' otherwise, signifying peak booking speed.
    """
    time_factor = np.log(1 + time_elapsed) if time_elapsed > 0 else 0
    rate = seats_sold / (1 + time_factor)
    low_threshold = num_seats / (4 * time_horizon)
    medium_threshold = (3 * num_seats) / (5 * time_horizon)
    high_threshold = num_seats / time_horizon

    if rate < low_threshold:
        return 'Low'
    elif rate < medium_threshold:
        return 'Medium'
    elif rate < high_threshold:
        return 'High'
    else:
        return 'High'

def get_competitor_price_level(time):
    """
    Assigns competitor pricing dynamically based on time progression and stochastic fluctuations.
    - `baseline_level`: A fixed sequence of competitor price levels ['Low', 'Medium', 'High'].
    - `shift_factor`: Adjusts the baseline index proportionally to the elapsed time.
    - `fluctuation`: Introduces random variation to the index, sampled from a normal distribution.
    - Ensures dynamic pricing with probabilistic variations, returning one of the baseline levels.
    """
    baseline_level = ['Low', 'Medium', 'High']
    shift_factor = (time / time_horizon) * len(baseline_level)
    fluctuation = np.random.normal(loc=0, scale=0.5)
    adjusted_level = int((shift_factor + fluctuation) % len(baseline_level))
    return baseline_level[adjusted_level]

def get_customer_segment(time, seats_left):
    """
    Randomly assigns customer segment, favoring business travelers as time approaches zero.
    - `economy_bias`: Represents the likelihood of an 'Economy' customer:
      - Decreases as time decreases (more last-minute bookings by business travelers).
    - Ensures higher probability of 'Business' customers when seats left are critically low (< num_seats / 3).
    - Randomly chooses between 'Economy' and 'Business' when conditions are neutral.
    """
    economy_bias = max(0.2, 1 - time / time_horizon)
    if seats_left < num_seats / 3:
        return 'Business' if np.random.rand() < (1 - economy_bias) else 'Economy'
    return random.choice(customer_segments)

# Q-learning parameters
alpha = 0.15  # Learning rate
gamma = 0.85  # Discount factor
epsilon = 0.2  # Exploration rate
episodes = 5000  # Number of training episodes, Change to number that can run it

states = []
for seats_left in range(num_seats + 1):
    for time in range(time_horizon + 1):
        for booking_rate in booking_rates:
            for competitor_price in competitor_prices:
                for customer_segment in customer_segments:
                    states.append((seats_left, time, booking_rate, competitor_price, customer_segment))

actions = price_levels
Q = {}
for state in states:
    for action in actions:
        Q[(state, action)] = 0.0

# Training the Q-learning model
for episode in range(episodes):
    seats_left = num_seats
    seats_sold = 0
    time = time_horizon - 1
    total_reward = 0
    time_elapsed = 0

    while time > 0 and seats_left > 0:
        booking_rate = get_booking_rate(seats_sold, time_elapsed)
        competitor_price_level = get_competitor_price_level(time)
        customer_segment = random.choice(customer_segments)
        state = (seats_left, time, booking_rate, competitor_price_level, customer_segment)

        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            q_values = [Q[(state, a)] for a in actions]
            max_q = max(q_values)
            best_actions = [a for a in actions if Q[(state, a)] == max_q]
            action = random.choice(best_actions)

        demand_probability = get_demand_probability(action, customer_segment, competitor_price_level)
        sale_occurred = np.random.binomial(1, demand_probability)

        if sale_occurred:
            seats_left -= 1
            seats_sold += 1
            reward = action
        else:
            reward = 0

        total_reward += reward

        time -= 1
        time_elapsed += 1

        next_booking_rate = get_booking_rate(seats_sold, time_elapsed)
        next_competitor_price_level = get_competitor_price_level(time)
        next_state = (seats_left, time, next_booking_rate, next_competitor_price_level, customer_segment)

        if next_state in states:
            future_q_values = [Q[(next_state, a)] for a in actions]
            max_future_q = max(future_q_values)
        else:
            max_future_q = 0

        Q[(state, action)] += alpha * (reward + gamma * max_future_q - Q[(state, action)])

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

optimal_policy = {} # Extract the optimal pricing policy based on trained Q-values
for state in states:
    q_values = [Q[(state, a)] for a in actions]
    max_q = max(q_values)
    best_actions = [a for a in actions if Q[(state, a)] == max_q]
    optimal_action = random.choice(best_actions)
    optimal_policy[state] = optimal_action

# print("\nOptimal pricing policy when time is 5 and customer segment is 'Economy':")
# for seats_left in range(num_seats + 1):
#     for booking_rate in booking_rates:
#         for competitor_price in competitor_prices:
#             state = (seats_left, 5, booking_rate, competitor_price, 'Economy')
#             action = optimal_policy.get(state, None)
#             if action:
#                 print(f"Seats left: {seats_left}, Booking rate: {booking_rate}, "
#                       f"Competitor price: {competitor_price}, Optimal price: {action}")
