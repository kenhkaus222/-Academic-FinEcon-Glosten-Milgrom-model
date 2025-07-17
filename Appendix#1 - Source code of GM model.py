# import modules to be used
import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_simulations = 10
num_orders = 100
theta = 0.5  # Probability that the true value is high
# pi = 0.3     # Probability that a trader is informed
v_h = 102    # High value
v_l = 98     # Low value
   
# Calculate ask_price and theta_a
def calculate_ask_price(mu_last, pi, theta_last, v_h, v_l):
    new_theta_a = ((1 + pi) / 2 * theta_last) / ((pi * theta_last) + ((1 - pi) / 2))
    new_ask_price = mu_last + ((((pi * theta_last) * (1 - theta_last))* (v_h - v_l)) / ((pi * theta_last) + ((1 - pi)/ 2))) 

    return new_theta_a, new_ask_price

# Calculate ask_price and theta_a
def calculate_bid_price(mu_last, pi, theta_last, v_h, v_l):
    new_theta_b = ((1 - pi) / 2 * theta_last) / ((pi * (1 - theta_last)) + ((1 - pi) / 2))
    new_bid_price = mu_last - (((pi * theta_last) * (1 - theta_last))* (v_h - v_l)) / ((pi * (1 - theta_last)) + ((1 - pi) / 2)) 
    
    return new_theta_b, new_bid_price

# Function to PD measure of one set of orders
def pd_measure(price): 
    return (price - v_h)**2

# Function to simulate one set of orders
def simulate_order(pi):
    # Generate 100 random orders
    orders = np.random.choice([1, -1], size=num_orders, p=[(1+pi)/2, (1-pi)/2])  # 1 for buy, -1 for sell with distribution under GM model
    beliefs = []  # To store updated beliefs
    prices = []  # To store prices

    # Initial belief and price
    belief = theta
    price = (v_h + v_l) / 2

    beliefs.append(belief)

    for order in orders:
        if order == 1:          # Buy order
            belief, price = calculate_ask_price(price, pi, belief, v_h, v_l)
        else:                   # Sell order
            belief, price = calculate_bid_price(price, pi, belief, v_h, v_l)

        beliefs.append(belief)
        prices.append(price)

    pds = [pd_measure(price) for price in prices]

    return beliefs, prices, pds ## Solution #1

def simulate_orders(pi, num_orders=num_simulations):
    all_beliefs = []
    all_prices = []
    all_PDs = []

    for _ in range(num_orders):
        beliefs, prices, pds = simulate_order(pi)
        all_beliefs.append(beliefs) 
        all_prices.append(prices)
        all_PDs.append(pds)

    # averaging PDs in one curve
    mean_process_PDs = np.mean(all_PDs, axis = 0)

    return all_beliefs, all_prices, all_PDs, mean_process_PDs
    

# Data Visualization for beliefs and PD measure
def plot_1_2():    
    # chart 1
    plt.figure(figsize=(12, 6))

    beliefs, prices, pds, mean_process_PDs = simulate_orders(0.3)

    for belief in beliefs:
        plt.plot(belief, marker='o', alpha = 0.5)

    plt.title('Belief Updates Over Time in each run under GM Model (10 runs)')
    plt.xlabel('Trade Number (1 to 100)')
    plt.ylabel('Belief (Theta)')
    plt.ylim(0, 1.2)
    plt.grid()

    # chart 2
    plt.figure(figsize=(12,6))

    for pd in pds:
        plt.plot(pd, marker='o', alpha = 0.5)

    plt.title('Pricing error: each curve shows the evolution of the pricing error in each run (10 runs)')
    plt.xlabel('Trade #')
    plt.ylim(0, 16)
    plt.grid()

# Data visualization for average PD measure with 3 different pi_s 
def plot_3():
    # chart 3
    plt.figure(figsize=(12, 6))

    pi_s = [0.1, 0.5, 0.9]
    for pi in pi_s:
        _, _, _, mean_pd = simulate_orders(pi)
        plt.plot(mean_pd, label=pi, marker='o', alpha = 0.5)
        
    plt.title('Squared pricing error, averaged over 10 simulations for pi = 10%, 50% & 90%')
    plt.xlabel('Trade #')
    plt.ylim(0,5)
    plt.legend(
        loc='best',
        title='π',
        title_fontsize=20)
    plt.grid()

# Execution of graph #1, #2 and #3
plot_1_2()
plot_3()
plt.show()
