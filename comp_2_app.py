import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

###############################################################################
# 1) Functions from your script
###############################################################################
def get_worst_decile_pool(returns_series, quantile_level=0.1):
    clean_rets = returns_series.dropna()
    threshold = clean_rets.quantile(quantile_level)
    worst_pool = clean_rets[clean_rets <= threshold]
    return worst_pool.values

def apply_shocks_in_blocks(sim_data, shock_pool, max_shock_events=5,
                           max_event_length=24, bias_toward_start=False):
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
            sim_data, shock_pool=shock_pool,
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

def compute_balance_paths(all_monthly_returns, inflation_draws,
                          withdrawal_schedule, initial_balance=100000,
                          ifa_fee_annual=0.0, apply_momentum=False,
                          momentum_window=10):
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
            else:
                annual_withdrawal = withdrawal_schedule[-1]
            monthly_withdrawal = annual_withdrawal / 12.0

            infl = inflation_draws[i, m]
            effective_monthly_return = sim_returns[m] - infl

            balance *= (1.0 + effective_monthly_return)
            balance -= monthly_withdrawal
            balance *= (1.0 - fee_monthly)
            balance_paths[i, m] = balance
    return balance_paths

def run_25yr_sim_and_plot(rets_df, inflation_data, portfolios,
                          withdrawal_csv_loc,
                          initial_balance=100000, n_sims=1000,
                          bootstrap_prob=1.0/3, max_shock_events=5,
                          max_event_length=24, bias_toward_start=False,
                          momentum_window=10, ifa_fee_benchmark=0.01,
                          ifa_fee_momentum=0.0125, starting_age=65,
                          quantiles_to_plot=[5,50,95]):

    #--------------------------------------------------------------------------
    # DEBUG INFO: Show rets_df columns, inflation_data shape
    #--------------------------------------------------------------------------
    st.write("[DEBUG] rets_df columns:", list(rets_df.columns))
    st.write("[DEBUG] rets_df shape:", rets_df.shape)
    st.write("[DEBUG] inflation_data shape:", inflation_data.shape)

    #--------------------------------------------------------------------------
    # Read withdrawal CSV
    #--------------------------------------------------------------------------
    try:
        withdrawal_df = pd.read_csv(withdrawal_csv_loc)
    except Exception as e:
        st.write("[DEBUG] Error reading CSV:", e)
        return

    st.write("[DEBUG] Original withdrawal_df shape:", withdrawal_df.shape)
    st.write(withdrawal_df.head(5))

    #--------------------------------------------------------------------------
    # Filter for ages >= starting_age
    #--------------------------------------------------------------------------
    st.write(f"[DEBUG] Filtering for age >= {starting_age}")
    withdrawal_df = withdrawal_df[withdrawal_df['age'] >= starting_age].sort_values(by='age')
    st.write("[DEBUG] After filter, withdrawal_df shape:", withdrawal_df.shape)
    st.write(withdrawal_df.head(5))

    # If no rows remain, we can't proceed
    if len(withdrawal_df) == 0:
        st.write(f"[DEBUG] No data for age >= {starting_age}. Simulation aborted.")
        return

    #--------------------------------------------------------------------------
    # Build withdrawal schedule
    #--------------------------------------------------------------------------
    withdrawal_schedule = withdrawal_df['withdrawal'].values
    horizon_years = len(withdrawal_schedule)
    horizon_months = horizon_years * 12

    st.write("[DEBUG] horizon_years:", horizon_years)
    st.write("[DEBUG] horizon_months:", horizon_months)

    #--------------------------------------------------------------------------
    # For each portfolio
    #--------------------------------------------------------------------------
    for port in portfolios:
        st.write("==========================================================")
        st.write(f"[DEBUG] Now simulating portfolio: {port}")
        if port not in rets_df.columns:
            st.write(f"[DEBUG] Portfolio '{port}' not found in rets_df columns! Skipping.")
            continue

        port_returns = rets_df[port].dropna()
        st.write(f"[DEBUG] port_returns length for '{port}':", len(port_returns))

        if len(port_returns) == 0:
            st.write(f"[DEBUG] No data in port_returns for '{port}' - skipping.")
            continue

        # Get shock pool
        shock_pool = get_worst_decile_pool(port_returns, 0.1)
        st.write(f"[DEBUG] shock_pool length for '{port}':", len(shock_pool))

        #--------------------------------------------------------------------------
        # Generate monthly returns with shocks
        #--------------------------------------------------------------------------
        if horizon_months <= 0:
            st.write("[DEBUG] Horizon months is 0 or less, skipping simulation.")
            continue

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
        st.write(f"[DEBUG] all_sim_monthly_returns shape for '{port}':", all_sim_monthly_returns.shape)

        #--------------------------------------------------------------------------
        # Random inflation draws
        #--------------------------------------------------------------------------
        inflation_array = np.asarray(inflation_data).flatten()
        st.write(f"[DEBUG] inflation_array length (after flatten): {len(inflation_array)}")
        if len(inflation_array) == 0:
            st.write("[DEBUG] No inflation data! Exiting.")
            return

        inflation_draws = np.random.choice(inflation_array, size=(n_sims, horizon_months))
        st.write("[DEBUG] inflation_draws shape:", inflation_draws.shape)

        #--------------------------------------------------------------------------
        # Compute baseline vs. momentum balance paths
        #--------------------------------------------------------------------------
        st.write("[DEBUG] Running compute_balance_paths for baseline...")
        baseline_balance_paths = compute_balance_paths(
            all_sim_monthly_returns,
            inflation_draws,
            withdrawal_schedule,
            initial_balance=initial_balance,
            ifa_fee_annual=ifa_fee_benchmark,
            apply_momentum=False
        )

        st.write("[DEBUG] Running compute_balance_paths for momentum...")
        momentum_balance_paths = compute_balance_paths(
            all_sim_monthly_returns,
            inflation_draws,
            withdrawal_schedule,
            initial_balance=initial_balance,
            ifa_fee_annual=ifa_fee_momentum,
            apply_momentum=True,
            momentum_window=momentum_window
        )

        st.write("[DEBUG] baseline_balance_paths shape:", baseline_balance_paths.shape)
        st.write("[DEBUG] momentum_balance_paths shape:", momentum_balance_paths.shape)

        #--------------------------------------------------------------------------
        # Build time/age axis
        #--------------------------------------------------------------------------
        time_axis = np.arange(horizon_months)
        age_axis = starting_age + time_axis / 12.0

        st.write("[DEBUG] age_axis (first 5):", age_axis[:5])

        #--------------------------------------------------------------------------
        # Calculate quantiles
        #--------------------------------------------------------------------------
        baseline_quantiles = {}
        momentum_quantiles = {}
        for q in quantiles_to_plot:
            baseline_quantiles[q] = np.percentile(baseline_balance_paths, q, axis=0)
            momentum_quantiles[q] = np.percentile(momentum_balance_paths, q, axis=0)

        #--------------------------------------------------------------------------
        # Plot
        #--------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10,6))
        for q in quantiles_to_plot:
            ax.plot(age_axis, baseline_quantiles[q],
                    label=f'Benchmark {q}th %ile', lw=2)
        for q in quantiles_to_plot:
            ax.plot(age_axis, momentum_quantiles[q],
                    label=f'Active {q}th %ile', lw=2, ls='--')

        ax.set_title(f'Portfolio: {port}\nFrom Age {starting_age} to {starting_age + horizon_years} (n={n_sims} sims)')
        ax.set_xlabel('Age')
        ax.set_ylabel('Balance')
        ax.grid(True)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))

        st.pyplot(fig)

###############################################################################
# 2) MAIN APP
###############################################################################
def main():
    st.title("Portfolio Simulation (Streamlit Cloud) - Debug Version")

    # Let the user upload .xlsx (with rets + inflation) and .csv (withdrawals).
    xlsx_file = st.file_uploader("Upload comparison_data.xlsx", type=["xlsx"])
    csv_file = st.file_uploader("Upload withdrawal_csv.csv", type=["csv"])

    if xlsx_file and csv_file:
        st.write("[DEBUG] Reading rets & inflation from uploaded Excel...")

        # Read returns data
        rets_df = pd.read_excel(xlsx_file, sheet_name='rets', index_col=0).dropna(axis=1, how='all')/100.0
        # Read inflation data
        inflation_df = pd.read_excel(xlsx_file, sheet_name='inflation', index_col=0).dropna(axis=1, how='all')/100.0

        # Show shapes, heads
        st.write("[DEBUG] rets_df shape:", rets_df.shape)
        st.write(rets_df.head(5))
        st.write("[DEBUG] inflation_df shape:", inflation_df.shape)
        st.write(inflation_df.head(5))

        # Hard-coded portfolios
        portfolios = [
            'IA Global',
            'IA Mixed Investment 0-35% Shares',
            'IA Mixed Investment 20-60% Shares',
            'IA Mixed Investment 40-85% Shares'
        ]

        st.write("[DEBUG] About to run run_25yr_sim_and_plot...")

        run_25yr_sim_and_plot(
            rets_df=rets_df,
            inflation_data=inflation_df,
            portfolios=portfolios,
            withdrawal_csv_loc=csv_file,  # read from user upload
            n_sims=2000,
            bootstrap_prob=1.0/3,
            max_shock_events=5,
            max_event_length=36,
            bias_toward_start=True,
            momentum_window=10,
            starting_age=58,
            quantiles_to_plot=[5, 50, 95],
            ifa_fee_benchmark=0.005,
            ifa_fee_momentum=0.0165,
            initial_balance=218714,
        )

    else:
        st.write("Please upload both the Excel file (rets + inflation) and CSV (withdrawals).")

if __name__ == "__main__":
    main()
