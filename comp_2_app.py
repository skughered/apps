import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

###############################################################################
# 1) Core Functions (Shocks, Momentum, Contributions)
###############################################################################
def get_worst_decile_pool(returns_series, quantile_level=0.1):
    clean_rets = returns_series.dropna()
    threshold = clean_rets.quantile(quantile_level)
    worst_pool = clean_rets[clean_rets <= threshold]
    return worst_pool.values

def apply_shocks_in_blocks(sim_data, shock_pool,
                           max_shock_events=5,
                           max_event_length=24,
                           bias_toward_start=False):
    horizon = len(sim_data)
    if shock_pool is None or len(shock_pool) == 0:
        return sim_data

    num_events = np.random.randint(1, max_shock_events + 1)
    for _ in range(num_events):
        event_length = np.random.randint(1, max_event_length + 1)
        if bias_toward_start:
            start_month = np.random.randint(0, max(horizon // 2, 1))
        else:
            start_month = np.random.randint(0, horizon)
        end_month = min(start_month + event_length, horizon)
        for t in range(start_month, end_month):
            sim_data[t] = np.random.choice(shock_pool)
    return sim_data

def stationary_bootstrap_sample(data, horizon, p):
    n = len(data)
    sample = []
    while len(sample) < horizon:
        start = np.random.randint(0, n)
        L = np.random.geometric(p)
        # Wrap-around if needed
        if start + L <= n:
            block = data[start:start+L]
        else:
            block = np.concatenate((data[start:], data[:(start + L - n)]))
        sample.extend(block.tolist())
    return np.array(sample[:horizon])

def stationary_bootstrap_with_shocks(returns, horizon=300, n_sims=1000,
                                     prob=1.0/3, shock_pool=None,
                                     max_shock_events=5, max_event_length=24,
                                     bias_toward_start=False):
    """
    Generate bootstrap samples of monthly returns, then apply shock events.
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
    If the sum of previous `window` months is negative, set current month's return to 0.
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
    For each month:
      - Optionally zero out returns if momentum < 0
      - Subtract inflation from monthly returns
      - Subtract monthly withdrawal
      - Add monthly contribution
      - Subtract monthly IFA fee
      => Return array of balances over time.
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

            balance *= (1.0 + effective_monthly_return)
            balance -= monthly_withdrawal
            balance += monthly_contribution
            balance *= (1.0 - fee_monthly)

            balance_paths[i, m] = balance

    return balance_paths

###############################################################################
# 2) Single-Portfolio Runner with Contributions (No Debug Statements)
###############################################################################
def run_25yr_sim_and_plot(
    portfolio_name,
    rets_df,
    inflation_df,
    withdrawal_csv_loc,
    initial_balance=100000,
    n_sims=1000,
    bootstrap_prob=1.0/3,
    max_shock_events=5,
    max_event_length=24,
    bias_toward_start=False,
    momentum_window=10,
    ifa_fee_benchmark=0.01,
    ifa_fee_momentum=0.0125,
    starting_age=65,
    quantiles_to_plot=[5, 50, 95]
):
    # Check portfolio in columns
    if portfolio_name not in rets_df.columns:
        st.error(f"Portfolio '{portfolio_name}' not found in rets_df columns.")
        return

    # Read CSV
    try:
        withdrawal_df = pd.read_csv(withdrawal_csv_loc)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    if not {"age","withdrawal","contribution"}.issubset(withdrawal_df.columns):
        st.error("CSV must have columns: age, withdrawal, contribution.")
        return

    # Filter for ages >= starting_age
    withdrawal_df = withdrawal_df[withdrawal_df['age'] >= starting_age].sort_values(by='age')
    if len(withdrawal_df) == 0:
        st.error(f"No rows after filtering for age >= {starting_age}.")
        return

    # Build schedules
    withdrawal_schedule = withdrawal_df["withdrawal"].values
    contribution_schedule = withdrawal_df["contribution"].values
    horizon_years = len(withdrawal_schedule)
    horizon_months = horizon_years * 12
    if horizon_months <= 0:
        st.error("Horizon months <= 0. Cannot run simulation.")
        return

    # Portfolio returns
    port_returns = rets_df[portfolio_name].dropna()
    shock_pool = get_worst_decile_pool(port_returns, 0.1)

    # (1) Extended horizon to handle the warm-up for momentum
    extended_horizon_months = horizon_months + momentum_window

    # (2) Generate monthly returns with shocks for the extended horizon
    all_sim_monthly_returns_extended = stationary_bootstrap_with_shocks(
        returns=port_returns,
        horizon=extended_horizon_months,
        n_sims=n_sims,
        prob=bootstrap_prob,
        shock_pool=shock_pool,
        max_shock_events=max_shock_events,
        max_event_length=max_event_length,
        bias_toward_start=bias_toward_start
    )

    # Positive-only inflation array
    inflation_data = inflation_df.dropna(axis=1, how="all") / 100.0
    inflation_array = np.asarray(inflation_data).flatten()
    inflation_array = inflation_array[inflation_array > 0]
    if len(inflation_array) == 0:
        st.error("No positive inflation data after filter.")
        return

    # (3) Random draws of inflation for the extended horizon
    inflation_draws_extended = np.random.choice(inflation_array, size=(n_sims, extended_horizon_months))

    # ------------------------------------------------------------------------
    # (4) Apply the momentum filter "externally" so that we can discard warm-up
    # ------------------------------------------------------------------------
    # 4a) Build a "momentum-filtered" copy
    momentum_filtered_returns_extended = np.zeros_like(all_sim_monthly_returns_extended)
    for i in range(n_sims):
        momentum_filtered_returns_extended[i, :] = apply_momentum_filter(
            all_sim_monthly_returns_extended[i, :], window=momentum_window
        )

    # 4b) Slice off the first 'momentum_window' months for both baseline & momentum
    #     => This ensures their official "Month 0" starts at the same time,
    #        so both are at the same initial balance in the first reported month.
    baseline_returns = all_sim_monthly_returns_extended[:, momentum_window:]
    momentum_returns = momentum_filtered_returns_extended[:, momentum_window:]
    inflation_draws = inflation_draws_extended[:, momentum_window:]

    # ------------------------------------------------------------------------
    # (5) Now pass only the final horizon_months to compute_balance_paths
    #     - Importantly, we set apply_momentum=False in BOTH calls, because
    #       we already applied the momentum filter above.
    # ------------------------------------------------------------------------
    baseline_balance_paths = compute_balance_paths(
        all_monthly_returns=baseline_returns,
        inflation_draws=inflation_draws,
        withdrawal_schedule=withdrawal_schedule,
        contribution_schedule=contribution_schedule,
        initial_balance=initial_balance,
        ifa_fee_annual=ifa_fee_benchmark,
        apply_momentum=False,       # Already handled "no momentum" scenario
        momentum_window=momentum_window
    )

    momentum_balance_paths = compute_balance_paths(
        all_monthly_returns=momentum_returns,
        inflation_draws=inflation_draws,
        withdrawal_schedule=withdrawal_schedule,
        contribution_schedule=contribution_schedule,
        initial_balance=initial_balance,
        ifa_fee_annual=ifa_fee_momentum,
        apply_momentum=False,       # We already did the momentum filter externally
        momentum_window=momentum_window
    )

    # Build age axis
    time_axis = np.arange(horizon_months)
    age_axis = starting_age + time_axis / 12.0

    # Quantiles
    baseline_quantiles = {}
    momentum_quantiles = {}
    for q in quantiles_to_plot:
        baseline_quantiles[q] = np.percentile(baseline_balance_paths, q, axis=0)
        momentum_quantiles[q] = np.percentile(momentum_balance_paths, q, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for q in quantiles_to_plot:
        ax.plot(age_axis, baseline_quantiles[q], label=f'Benchmark {q}th %ile', lw=2)
    for q in quantiles_to_plot:
        ax.plot(age_axis, momentum_quantiles[q], label=f'Active {q}th %ile', lw=2, ls='--')

    ax.set_title(
        f"Portfolio: {portfolio_name}\n"
        f"From Age {starting_age} to {starting_age + horizon_years} (n={n_sims} sims)"
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Balance")
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}k"))
    st.pyplot(fig)



###############################################################################
# 3) MAIN STREAMLIT APP
###############################################################################
def main():
    st.title("Portfolio Simulation")

    # Read local Excel for returns & inflation
    try:
        rets_df = pd.read_excel("comparison_data.xlsx", sheet_name="rets", index_col=0)
        rets_df = rets_df.dropna(axis=1, how="all") / 100.0
        inflation_df = pd.read_excel("comparison_data.xlsx", sheet_name="inflation", index_col=0)
    except Exception as e:
        st.error(f"Error reading 'comparison_data.xlsx': {e}")
        return

    # User uploads CSV
    csv_file = st.file_uploader("Upload your CSV (age, withdrawal, contribution)", type=["csv"])
    if not csv_file:
        st.info("Please upload a CSV with columns: age, withdrawal, contribution.")
        return

    # Single portfolio selection
    portfolio_list = list(rets_df.columns)
    selected_portfolio = st.selectbox("Select Portfolio", portfolio_list)

    # Additional user inputs
    initial_balance = st.number_input("Initial Balance", value=300000, step=10000)
    benchmark_fee = st.number_input("Benchmark Fee (annual, decimal)", value=0.01, step=0.001, format="%.4f")
    active_fee = st.number_input("Active Fee (annual, decimal)", value=0.0125, step=0.001, format="%.4f")
    starting_age = st.number_input("Starting Age", value=56, step=1)
    n_sims = st.number_input("Number of Simulations", value=1000, step=500)

    run_button = st.button("Run Simulation")

    if run_button:
        run_25yr_sim_and_plot(
            portfolio_name=selected_portfolio,
            rets_df=rets_df,
            inflation_df=inflation_df,
            withdrawal_csv_loc=csv_file,
            initial_balance=initial_balance,
            n_sims=n_sims,
            bootstrap_prob=1.0/3,
            max_shock_events=3,
            max_event_length=12,
            bias_toward_start=False,
            momentum_window=10,
            ifa_fee_benchmark=benchmark_fee,
            ifa_fee_momentum=active_fee,
            starting_age=starting_age,
            quantiles_to_plot=[5, 50, 95]
        )

if __name__ == "__main__":
    main()
