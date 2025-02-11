import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

###############################################################################
# 1) Core Functions
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
      - If momentum < 0 (past window), returns[m] => 0 (optionally).
      - Subtract inflation from return each month.
      - Subtract monthly withdrawal, add monthly contribution.
      - Subtract monthly fee (ifa_fee_annual / 12).
    """
    n_sims, horizon = all_monthly_returns.shape
    balance_paths = np.zeros((n_sims, horizon))
    fee_monthly = ifa_fee_annual / 12.0

    for i in range(n_sims):
        if apply_momentum:
            sim_returns = apply_momentum_filter(
                all_monthly_returns[i, :], window=momentum_window
            )
        else:
            sim_returns = all_monthly_returns[i, :]

        balance = initial_balance
        for m in range(horizon):
            year_index = m // 12
            if year_index < len(withdrawal_schedule):
                annual_withdrawal = withdrawal_schedule[year_index]
                annual_contribution = contribution_schedule[year_index]
            else:
                # If horizon extends beyond CSV length, use the last row's values
                annual_withdrawal = withdrawal_schedule[-1]
                annual_contribution = contribution_schedule[-1]

            monthly_withdrawal = annual_withdrawal / 12.0
            monthly_contribution = annual_contribution / 12.0

            infl = inflation_draws[i, m]  # monthly inflation (decimal, e.g. 0.002 => 0.2%)
            effective_monthly_return = sim_returns[m] - infl

            # Apply returns
            balance *= (1.0 + effective_monthly_return)

            # Withdraw, then contribute
            balance -= monthly_withdrawal
            balance += monthly_contribution

            # Subtract monthly fee
            balance *= (1.0 - fee_monthly)

            balance_paths[i, m] = balance

    return balance_paths


###############################################################################
# 2) Single-Portfolio Simulation with Debug
###############################################################################
def run_25yr_sim_and_plot(
    portfolio_name,
    rets_df,
    inflation_data,
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
    """
    Runs a single-portfolio simulation with contributions and momentum option.
    Debug statements included (st.write).
    """
    #-----------------------------------------------------------------
    # 1) Check that portfolio_name is in rets_df
    #-----------------------------------------------------------------
    st.write("[DEBUG] Checking portfolio_name in rets_df columns...")
    if portfolio_name not in rets_df.columns:
        st.write(f"[DEBUG] Portfolio '{portfolio_name}' not found in rets_df columns. Aborting.")
        return

    #-----------------------------------------------------------------
    # 2) Read CSV (with age, withdrawal, contribution)
    #-----------------------------------------------------------------
    try:
        withdrawal_df = pd.read_csv(withdrawal_csv_loc)
        st.write("[DEBUG] withdrawal_df shape (pre-filter):", withdrawal_df.shape)
        st.write(withdrawal_df.head(5))
    except Exception as e:
        st.write("[DEBUG] Failed to read CSV:", e)
        return

    # Must have columns age, withdrawal, contribution
    required_cols = {'age','withdrawal','contribution'}
    if not required_cols.issubset(withdrawal_df.columns):
        st.write("[DEBUG] CSV missing required columns:", required_cols)
        return

    # Filter for ages >= starting_age
    st.write(f"[DEBUG] Filtering for age >= {starting_age} ...")
    withdrawal_df = withdrawal_df[withdrawal_df['age'] >= starting_age].sort_values(by='age')
    st.write("[DEBUG] withdrawal_df shape (post-filter):", withdrawal_df.shape)
    st.write(withdrawal_df.head(5))

    if len(withdrawal_df) == 0:
        st.write(f"[DEBUG] No rows remain after filtering for age >= {starting_age}. Aborting.")
        return

    #-----------------------------------------------------------------
    # 3) Build schedules from CSV
    #-----------------------------------------------------------------
    withdrawal_schedule = withdrawal_df['withdrawal'].values
    contribution_schedule = withdrawal_df['contribution'].values

    horizon_years = len(withdrawal_schedule)
    horizon_months = horizon_years * 12
    st.write("[DEBUG] horizon_years:", horizon_years)
    st.write("[DEBUG] horizon_months:", horizon_months)

    if horizon_months <= 0:
        st.write("[DEBUG] Horizon is zero. Aborting simulation.")
        return

    #-----------------------------------------------------------------
    # 4) Prepare the portfolio returns & shock pool
    #-----------------------------------------------------------------
    port_returns = rets_df[portfolio_name].dropna()
    st.write(f"[DEBUG] {portfolio_name}: port_returns length:", len(port_returns))

    shock_pool = get_worst_decile_pool(port_returns, 0.1)
    st.write("[DEBUG] shock_pool length:", len(shock_pool))

    #-----------------------------------------------------------------
    # 5) Stationary Bootstrap with shocks
    #-----------------------------------------------------------------
    st.write("[DEBUG] Generating all_sim_monthly_returns via stationary bootstrap...")
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
    st.write("[DEBUG] all_sim_monthly_returns shape:", all_sim_monthly_returns.shape)

    #-----------------------------------------------------------------
    # 6) Filter inflation data for positivity & generate draws
    #-----------------------------------------------------------------
    st.write("[DEBUG] inflation_data shape (raw):", inflation_data.shape)
    # Flatten, remove negative
    inflation_array = np.asarray(inflation_data).flatten()
    inflation_array = inflation_array[inflation_array > 0]  # keep only positive
    st.write("[DEBUG] inflation_array length (positive only):", len(inflation_array))

    if len(inflation_array) == 0:
        st.write("[DEBUG] No positive inflation data left. Aborting.")
        return

    inflation_draws = np.random.choice(inflation_array, size=(n_sims, horizon_months))
    st.write("[DEBUG] inflation_draws shape:", inflation_draws.shape)

    #-----------------------------------------------------------------
    # 7) Compute Baseline vs. Momentum
    #-----------------------------------------------------------------
    st.write("[DEBUG] Compute baseline balance_paths...")
    baseline_balance_paths = compute_balance_paths(
        all_sim_monthly_returns,
        inflation_draws,
        withdrawal_schedule,
        contribution_schedule,
        initial_balance=initial_balance,
        ifa_fee_annual=ifa_fee_benchmark,
        apply_momentum=False
    )

    st.write("[DEBUG] Compute momentum balance_paths...")
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

    st.write("[DEBUG] baseline_balance_paths shape:", baseline_balance_paths.shape)
    st.write("[DEBUG] momentum_balance_paths shape:", momentum_balance_paths.shape)

    #-----------------------------------------------------------------
    # 8) Build Age Axis & Plot
    #-----------------------------------------------------------------
    time_axis = np.arange(horizon_months)
    age_axis = starting_age + time_axis / 12.0
    st.write("[DEBUG] age_axis (first 5):", age_axis[:5])

    # Compute quantiles
    baseline_quantiles = {}
    momentum_quantiles = {}
    for q in quantiles_to_plot:
        baseline_quantiles[q] = np.percentile(baseline_balance_paths, q, axis=0)
        momentum_quantiles[q] = np.percentile(momentum_balance_paths, q, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10,6))
    for q in quantiles_to_plot:
        ax.plot(age_axis, baseline_quantiles[q],
                label=f'Benchmark {q}th %ile', lw=2)
    for q in quantiles_to_plot:
        ax.plot(age_axis, momentum_quantiles[q],
                label=f'Active {q}th %ile', lw=2, ls='--')

    ax.set_title(f'Portfolio: {portfolio_name}\nFrom Age {starting_age} to {starting_age + horizon_years} (n={n_sims} sims)')
    ax.set_xlabel('Age')
    ax.set_ylabel('Portfolio Balance')
    ax.grid(True)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))

    st.pyplot(fig)


###############################################################################
# 3) Streamlit MAIN
###############################################################################
def main():
    st.title("Single Portfolio Simulation with Contributions (Debug)")

    #--------------------------------------------------------
    # 3A) Read static returns & inflation from local Excel
    #     (No user upload for these)
    #--------------------------------------------------------
    try:
        rets_df = pd.read_excel("comparison_data.xlsx", sheet_name="rets", index_col=0)
        rets_df = rets_df.dropna(axis=1, how="all") / 100.0
        st.write("[DEBUG] rets_df shape:", rets_df.shape)
        st.write(rets_df.head(5))

        inflation_df = pd.read_excel("comparison_data.xlsx", sheet_name="inflation", index_col=0)
        inflation_df = inflation_df.dropna(axis=1, how="all") / 100.0
        st.write("[DEBUG] inflation_df shape:", inflation_df.shape)
        st.write(inflation_df.head(5))
    except Exception as e:
        st.write("[DEBUG] Failed to read local comparison_data.xlsx:", e)
        return

    #--------------------------------------------------------
    # 3B) Let user upload the CSV with age, withdrawal, contribution
    #--------------------------------------------------------
    csv_file = st.file_uploader("Upload your withdrawal+contribution CSV", type=["csv"])
    if not csv_file:
        st.info("Please upload a CSV with columns: age, withdrawal, contribution.")
        return

    #--------------------------------------------------------
    # 3C) Single-Portfolio Drop-Down
    #--------------------------------------------------------
    portfolio_list = list(rets_df.columns)  # or define a subset
    selected_portfolio = st.selectbox("Select Portfolio to Simulate:", portfolio_list)

    #--------------------------------------------------------
    # 3D) User Inputs: Initial Balance, Fees, Starting Age, # of Sims
    #--------------------------------------------------------
    initial_balance = st.number_input("Initial Balance", value=100000, step=1000)
    benchmark_fee = st.number_input("Benchmark Fee (annual, decimal)", value=0.01, step=0.001, format="%.4f")
    active_fee = st.number_input("Active (Momentum) Fee (annual, decimal)", value=0.0125, step=0.001, format="%.4f")
    starting_age = st.number_input("Starting Age", value=58, step=1)
    n_sims = st.number_input("Number of Simulations", value=1000, step=500)

    # Add a "Run Simulation" button
    run_button = st.button("Run Simulation")

    if run_button:
        st.write("[DEBUG] Starting run_25yr_sim_and_plot with user inputs:")
        st.write(f"   Portfolio: {selected_portfolio}")
        st.write(f"   Initial Balance: {initial_balance}")
        st.write(f"   Benchmark Fee: {benchmark_fee}")
        st.write(f"   Active Fee: {active_fee}")
        st.write(f"   Starting Age: {starting_age}")
        st.write(f"   n_sims: {n_sims}")

        run_25yr_sim_and_plot(
            portfolio_name=selected_portfolio,
            rets_df=rets_df,
            inflation_data=inflation_df,  # We'll remove negatives in function
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
