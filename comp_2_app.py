import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

###############################################################################
# 1) Helper Functions
###############################################################################
def get_worst_decile_pool(returns_series, quantile_level=0.1):
    """
    Return an array of returns that lie in the worst quantile_level fraction
    (e.g., the worst 10% if quantile_level=0.1).
    """
    clean_rets = returns_series.dropna()
    threshold = clean_rets.quantile(quantile_level)
    worst_pool = clean_rets[clean_rets <= threshold]
    return worst_pool.values

def apply_shocks_in_blocks(sim_data, shock_pool,
                           max_shock_events=5,
                           max_event_length=24,
                           bias_toward_start=False):
    """
    Inject multiple 'shock events' into sim_data. Each event is up to max_event_length months,
    replaced with random draws from shock_pool.
    """
    horizon = len(sim_data)
    if shock_pool is None or len(shock_pool) == 0:
        return sim_data  # no shocks if shock_pool is empty

    # Randomly decide how many shock events to apply (1 to max_shock_events)
    num_events = np.random.randint(1, max_shock_events + 1)
    for _ in range(num_events):
        event_length = np.random.randint(1, max_event_length + 1)
        # Pick start month (biasing toward the start if desired)
        if bias_toward_start:
            start_month = np.random.randint(0, max(horizon // 2, 1))
        else:
            start_month = np.random.randint(0, horizon)
        end_month = min(start_month + event_length, horizon)
        # Replace that block with random draws from the worst pool
        for t in range(start_month, end_month):
            sim_data[t] = np.random.choice(shock_pool)
    return sim_data

def stationary_bootstrap_sample(data, horizon, p):
    """
    Manually generate a stationary bootstrap sample of length `horizon`
    from the original data using probability p for ending blocks.
    """
    n = len(data)
    sample = []
    while len(sample) < horizon:
        start = np.random.randint(0, n)
        # geometric distribution block length
        L = np.random.geometric(p)
        # Wrap around if needed
        if start + L <= n:
            block = data[start:start + L]
        else:
            block = np.concatenate((data[start:], data[:(start + L - n)]))
        sample.extend(block.tolist())
    return np.array(sample[:horizon])

def stationary_bootstrap_with_shocks(returns,
                                     horizon=300,
                                     n_sims=1000,
                                     prob=1.0/3,
                                     shock_pool=None,
                                     max_shock_events=5,
                                     max_event_length=24,
                                     bias_toward_start=False):
    """
    Generate bootstrap samples of monthly returns of length `horizon`,
    then apply 'shock events' to each path.
    """
    returns = np.asarray(returns).flatten()
    all_sim_monthly_returns = np.zeros((n_sims, horizon))
    for i in range(n_sims):
        sim_data = stationary_bootstrap_sample(returns, horizon, prob)
        sim_data = apply_shocks_in_blocks(
            sim_data,
            shock_pool=shock_pool,
            max_shock_events=max_shock_events,
            max_event_length=max_event_length,
            bias_toward_start=bias_toward_start
        )
        all_sim_monthly_returns[i, :] = sim_data
    return all_sim_monthly_returns

def apply_momentum_filter(returns, window=10):
    """
    If the sum of returns over the past `window` months is negative,
    set the current month's return to 0.
    """
    filtered = returns.copy()
    for m in range(window, len(returns)):
        momentum = np.sum(returns[m - window:m])
        if momentum < 0:
            filtered[m] = 0.0
    return filtered

def compute_balance_paths(all_monthly_returns,
                          inflation_draws,
                          withdrawal_schedule,
                          contribution_schedule,
                          initial_balance=100000,
                          ifa_fee_annual=0.0,
                          apply_momentum=False,
                          momentum_window=10):
    """
    - all_monthly_returns: (n_sims x horizon) array of monthly returns
    - inflation_draws: (n_sims x horizon) array of monthly inflation
    - withdrawal_schedule & contribution_schedule: annual amounts, each array has length = # of years
    - For each month:
      1) Adjust balance by monthly return minus monthly inflation.
      2) Subtract monthly withdrawal, add monthly contribution.
      3) Subtract monthly fee.
    Returns (n_sims x horizon) array of end-of-month balances.
    """
    n_sims, horizon = all_monthly_returns.shape
    balance_paths = np.zeros((n_sims, horizon))
    fee_monthly = ifa_fee_annual / 12.0

    for i in range(n_sims):
        if apply_momentum:
            sim_returns = apply_momentum_filter(all_monthly_returns[i, :], window=momentum_window)
        else:
            sim_returns = all_monthly_returns[i, :]

        balance = initial_balance
        for m in range(horizon):
            year_index = m // 12
            # If the simulation goes beyond the last year, repeat the final withdrawal/contribution
            if year_index < len(withdrawal_schedule):
                annual_withdrawal = withdrawal_schedule[year_index]
                annual_contribution = contribution_schedule[year_index]
            else:
                annual_withdrawal = withdrawal_schedule[-1]
                annual_contribution = contribution_schedule[-1]

            monthly_withdrawal = annual_withdrawal / 12.0
            monthly_contribution = annual_contribution / 12.0

            infl = inflation_draws[i, m]
            effective_monthly_return = sim_returns[m] - infl

            # Growth from returns
            balance *= (1.0 + effective_monthly_return)

            # Withdraw, then contribute
            balance -= monthly_withdrawal
            balance += monthly_contribution

            # Subtract monthly fee
            balance *= (1.0 - fee_monthly)

            balance_paths[i, m] = balance

    return balance_paths

def run_25yr_sim_and_plot(
        rets_df,
        inflation_data,
        portfolio_name,
        withdrawal_csv_loc,
        initial_balance=100000,
        n_sims=1000,
        bootstrap_prob=1.0 / 3,
        max_shock_events=5,
        max_event_length=24,
        bias_toward_start=False,
        momentum_window=10,
        ifa_fee_benchmark=0.01,
        ifa_fee_momentum=0.0125,
        starting_age=65,
        quantiles_to_plot=[5, 50, 95]
):
    """
    Runs the simulation for a SINGLE portfolio, using:
      - returns from rets_df[portfolio_name]
      - inflation_data (numpy array or DataFrame) => random draws
      - withdrawal CSV with columns: age, withdrawal, contribution
      - user-defined fees, initial balance, etc.

    Produces two sets of result lines (Benchmark vs. Active with momentum)
    and plots them on a single figure.
    """
    # 1) Read the withdrawal CSV
    withdrawal_df = pd.read_csv(withdrawal_csv_loc)
    # Filter for ages >= starting_age, then sort
    withdrawal_df = withdrawal_df[withdrawal_df['age'] >= starting_age].sort_values(by='age')

    # Convert to arrays
    withdrawal_schedule = withdrawal_df['withdrawal'].values
    contribution_schedule = withdrawal_df['contribution'].values

    # Determine horizon (in years) from the schedule length
    horizon_years = len(withdrawal_schedule)
    horizon_months = horizon_years * 12

    # 2) Prepare returns + shock pool
    port_returns = rets_df[portfolio_name].dropna()
    shock_pool = get_worst_decile_pool(port_returns, 0.1)

    # 3) Generate simulated monthly returns
    all_sim_monthly_returns = stationary_bootstrap_with_shocks(
        returns=port_returns,
        horizon=horizon_months,
        n_sims=n_sims,
        prob=bootstrap_prob,
        shock_pool=shock_pool,
        max_shock_events=max_shock_events,
        max_event_length=max_event_length,
        bias_toward_start=bias_toward_start
    )

    # 4) Generate random draws of inflation (positive-only data)
    inflation_array = np.asarray(inflation_data).flatten()
    # If it's a DataFrame, it's likely Nx1 or NxM, so flatten
    # We already removed negative values in the main code. Just proceed:
    inflation_draws = np.random.choice(inflation_array, size=(n_sims, horizon_months))

    # 5) Balance paths for Benchmark vs. Momentum
    baseline_balance_paths = compute_balance_paths(
        all_sim_monthly_returns,
        inflation_draws,
        withdrawal_schedule,
        contribution_schedule,
        initial_balance=initial_balance,
        ifa_fee_annual=ifa_fee_benchmark,
        apply_momentum=False
    )

    momentum_balance_paths = compute_balance_paths(
        all_sim_monthly_returns,
        inflation_draws,
        withdrawal_schedule,
        contribution_schedule,
        initial_balance=initial_balance,
        ifa_fee_annual=ifa_fee_momentum,
        apply_momentum=True,
        momentum_window=momentum_window
    )

    # 6) Compute and Plot
    time_axis = np.arange(horizon_months)
    age_axis = starting_age + time_axis / 12.0

    baseline_quantiles = {}
    momentum_quantiles = {}
    for q in quantiles_to_plot:
        baseline_quantiles[q] = np.percentile(baseline_balance_paths, q, axis=0)
        momentum_quantiles[q] = np.percentile(momentum_balance_paths, q, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot baseline cones
    for q in quantiles_to_plot:
        ax.plot(age_axis, baseline_quantiles[q],
                label=f'Benchmark {q}th %ile', lw=2)
    # Plot momentum cones (dashed)
    for q in quantiles_to_plot:
        ax.plot(age_axis, momentum_quantiles[q],
                label=f'Active {q}th %ile', lw=2, ls='--')

    ax.set_title(f'Portfolio: {portfolio_name}\nBalances from Age {starting_age} to {starting_age + horizon_years} (n={n_sims} sims)')
    ax.set_xlabel('Age')
    ax.set_ylabel('Portfolio Balance')
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))

    st.pyplot(fig)

###############################################################################
# 2) MAIN STREAMLIT APP
###############################################################################
def main():
    st.title("Portfolio Simulation with Contributions")

    # ---------------------------
    # 2A) Read static returns data from local file (no upload)
    # ---------------------------
    rets_df = pd.read_excel(
        "comparison_data.xlsx", sheet_name="rets", index_col=0
    ).dropna(axis=1, how='all') / 100.0

    # ---------------------------
    # 2B) Read inflation data from local file, remove negatives
    # ---------------------------
    inflation_df = pd.read_excel(
        "comparison_data.xlsx", sheet_name="inflation", index_col=0
    ).dropna(axis=1, how='all') / 100.0

    # Convert to 1D array
    inflation_array = np.asarray(inflation_df).flatten()
    # Remove all negative inflation values
    inflation_array = inflation_array[inflation_array > 0]
    # Create a small DataFrame so we can pass the same shape to the function
    # (Alternatively, we could just pass inflation_array directly)
    inflation_data = pd.DataFrame(inflation_array, columns=["inflation"])

    # ---------------------------
    # 2C) Let user upload the "withdrawal_csv.csv" with columns:
    #     age,withdrawal,contribution
    # ---------------------------
    csv_file = st.file_uploader("Upload your Withdrawal/Contribution CSV", type=["csv"])
    if not csv_file:
        st.info("Please upload a CSV with columns: age, withdrawal, contribution.")
        return

    # ---------------------------
    # 2D) User Inputs for Simulation
    # ---------------------------
    portfolio_list = [
        'IA Global',
        'IA Mixed Investment 0-35% Shares',
        'IA Mixed Investment 20-60% Shares',
        'IA Mixed Investment 40-85% Shares'
    ]
    selected_portfolio = st.selectbox("Select Portfolio:", portfolio_list)

    initial_balance = st.number_input("Initial Balance", value=100000, step=1000)
    benchmark_fee = st.number_input("Benchmark Fee (annual, decimal)", value=0.01, step=0.001, format="%.4f")
    active_fee = st.number_input("Active Fee (annual, decimal)", value=0.0125, step=0.001, format="%.4f")

    # Additional simulation parameters (you can expose more if desired)
    starting_age = st.number_input("Starting Age", value=58, step=1)
    n_sims = st.number_input("Number of Simulations", value=1000, step=500)
    run_button = st.button("Run Simulation")

    if run_button:
        st.write("Running Simulation...")
        # Here we call the main function to do all the steps
        run_25yr_sim_and_plot(
            rets_df=rets_df,
            inflation_data=inflation_data,
            portfolio_name=selected_portfolio,
            withdrawal_csv_loc=csv_file,
            initial_balance=initial_balance,
            n_sims=n_sims,
            bootstrap_prob=1.0/3,
            max_shock_events=5,
            max_event_length=36,
            bias_toward_start=True,
            momentum_window=10,
            ifa_fee_benchmark=benchmark_fee,
            ifa_fee_momentum=active_fee,
            starting_age=starting_age,
            quantiles_to_plot=[5, 50, 95]
        )

if __name__ == "__main__":
    main()
