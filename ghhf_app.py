import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Tuple, Dict

# -----------------------------
# Helpers
# -----------------------------

def aud(x: float) -> str:
    return f"${x:,.0f}"

def format_currency_compact(x: float) -> str:
    """Format currency with k for thousands, m for millions, etc."""
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}m"
    elif abs(x) >= 1_000:
        return f"${x/1_000:.0f}k"
    else:
        return f"${x:,.0f}"

@st.cache_data(show_spinner=False)
def simulate_paths(
    mu_annual: float,
    sigma_annual: float,
    years: int,
    sims: int,
    start_value: float,
    monthly_contrib: float,
    fee_annual: float,
    seed: int | None,
) -> np.ndarray:
    """Simulate end values using geometric Brownian motion with monthly steps.

    Returns an array of shape (sims,) for final portfolio values.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    months = years * 12
    # Convert annual params to monthly GBM parameters
    mu_m = mu_annual
    sig_m = sigma_annual

    drift = (mu_m - 0.5 * sig_m**2) / 12.0
    vol = sig_m / np.sqrt(12.0)

    # Random shocks for all sims & months
    shocks = rng.normal(0.0, 1.0, size=(sims, months))
    # Monthly gross return from GBM
    gross_returns = np.exp(drift + vol * shocks)

    # Apply fees as a monthly drag (Management Expense Ratio style)
    fee_factor = (1.0 - fee_annual / 12.0)

    # Iterate wealth path with contributions at the *start* of the month (conservative for fees)
    wealth = np.full(sims, start_value, dtype=float)
    for m in range(months):
        wealth += monthly_contrib
        wealth *= gross_returns[:, m]
        wealth *= fee_factor
    return wealth


def run_dual_sim(
    params_a: Dict[str, float],
    params_b: Dict[str, float],
    years: int,
    sims: int,
    start_value: float,
    monthly_contrib: float,
    seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run two Monte Carlo sets with the *same* random numbers for fairness.
    We accomplish this by generating one matrix of shocks and reusing it.
    """
    # Build a base set of shocks and reuse inside simulate-like loop
    rng = np.random.default_rng(seed)
    months = years * 12
    shocks = rng.normal(0.0, 1.0, size=(sims, months))

    def simulate_with_shocks(mu_annual: float, sigma_annual: float, fee_annual: float) -> np.ndarray:
        drift = (mu_annual - 0.5 * sigma_annual**2) / 12.0
        vol = sigma_annual / np.sqrt(12.0)
        gross_returns = np.exp(drift + vol * shocks)
        fee_factor = (1.0 - fee_annual / 12.0)
        wealth = np.full(sims, start_value, dtype=float)
        for m in range(months):
            wealth += monthly_contrib
            wealth *= gross_returns[:, m]
            wealth *= fee_factor
        return wealth

    a_final = simulate_with_shocks(params_a["mu"], params_a["sigma"], params_a["fee"])  # type: ignore[index]
    b_final = simulate_with_shocks(params_b["mu"], params_b["sigma"], params_b["fee"])  # type: ignore[index]
    return a_final, b_final

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="GHHF vs DHHF Monte Carlo", layout="wide")

with st.sidebar:
    st.header("Simulation Inputs")
    years = st.number_input("Years", min_value=1, max_value=60, value=20, step=1)
    sims = st.number_input("Simulations", min_value=500, max_value=200_000, value=20_000, step=500)
    start_value = st.number_input("Starting investment (A$)", min_value=0.0, value=50_000.0, step=1_000.0, format="%0.0f")
    monthly_contrib = st.number_input("Monthly contribution (A$)", min_value=0.0, value=1_000.0, step=100.0, format="%0.0f")
    seed_opt = st.text_input("Random seed (optional)", value="42")
    seed = int(seed_opt) if seed_opt.strip().isdigit() else None

    st.divider()
    st.subheader("Assumptions (annual, nominal)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**GHHF**")
        ghhf_mu = st.number_input("GHHF expected return %", min_value=-10.0, max_value=20.0, value=15.0, step=0.1, help="Your long-run nominal total return assumption") / 100.0
        ghhf_sig = st.number_input("GHHF volatility %", min_value=1.0, max_value=50.0, value=22.0, step=0.5, help="Annualised standard deviation") / 100.0
        
        st.markdown("*Fee structure:*")
        ghhf_mer = st.number_input("GHHF MER %", min_value=0.0, max_value=3.0, value=0.6, step=0.05, help="Management Expense Ratio") / 100.0
        ghhf_leverage = st.number_input("GHHF leverage cost %", min_value=0.0, max_value=5.0, value=2.0, step=0.1, help="Interest cost on leverage") / 100.0
        ghhf_fee = ghhf_mer + ghhf_leverage  # Total fee
        
        # Display total fee
        st.markdown(f"**Total fee: {ghhf_fee*100:.2f}%**")
        
    with col2:
        st.markdown("**DHHF**")
        dhhf_mu = st.number_input("DHHF expected return %", min_value=-10.0, max_value=20.0, value=11.0, step=0.1) / 100.0
        dhhf_sig = st.number_input("DHHF volatility %", min_value=1.0, max_value=50.0, value=16.0, step=0.5) / 100.0
        
        st.markdown("*Fee structure:*")
        dhhf_mer = st.number_input("DHHF MER %", min_value=0.0, max_value=2.0, value=0.19, step=0.01, help="Management Expense Ratio") / 100.0
        dhhf_leverage = st.number_input("DHHF leverage cost %", min_value=0.0, max_value=3.0, value=0.0, step=0.05, help="Interest cost on leverage") / 100.0
        dhhf_fee = dhhf_mer + dhhf_leverage  # Total fee
        
        # Display total fee
        st.markdown(f"**Total fee: {dhhf_fee*100:.2f}%**")

    st.divider()
    st.subheader("Inflation Adjustment")
    show_real_values = st.checkbox("Show real values (inflation-adjusted)", value=False, 
                                  help="Convert nominal returns to real returns by removing inflation")
    inflation_rate = st.number_input("Inflation rate %", min_value=0.0, max_value=10.0, value=3.0, step=0.1, 
                                    help="Annual inflation rate to remove from returns") / 100.0
    
    st.divider()
    st.subheader("Percentiles to display")
    default_pct = "1,5,10,25,50,75,90,95,99"
    pct_str = st.text_input("Comma-separated percentiles (0-100)", value=default_pct)

# Add title and caption after sidebar inputs are defined
col_title, col_button = st.columns([3, 1])
with col_title:
    st.title("📈 GHHF vs DHHF – Monte Carlo Simulator")
    if show_real_values:
        st.caption(f"Compare real (inflation-adjusted) terminal values via Monte Carlo. Inflation rate: {inflation_rate*100:.1f}% p.a.")
    else:
        st.caption("Compare nominal terminal values via Monte Carlo with stacked histograms and percentile bars.")

with col_button:
    st.write("")  # Add some spacing to align with title
    st.write("")  # Add more spacing
    update_button = st.button("🚀 Update Simulation", type="primary", use_container_width=True)
    
    # Show success message in the same column when simulation completes
    if 'simulation_success' in st.session_state and st.session_state.simulation_success:
        st.success("✅ Simulation completed successfully!")
        # Reset the flag after showing the message
        st.session_state.simulation_success = False
    
    # Show initial message when no simulation has been run yet
    if 'ghhf_final' not in st.session_state or st.session_state.ghhf_final is None:
        st.info("🚀 Click the 'Update Simulation' button above to run your first simulation!")

# Parse percentiles
try:
    percentiles = [p for p in sorted({max(0, min(100, int(x.strip())) ) for x in pct_str.split(',') if x.strip() != ''})]
    if len(percentiles) == 0:
        percentiles = [1,5,10,25,50,75,90,95,99]
except Exception:
    percentiles = [1,5,10,25,50,75,90,95,99]

# Apply inflation adjustment if real values are requested
if show_real_values:
    ghhf_mu_real = ghhf_mu - inflation_rate
    dhhf_mu_real = dhhf_mu - inflation_rate
    params_ghhf = {"mu": ghhf_mu_real, "sigma": ghhf_sig, "fee": ghhf_fee}
    params_dhhf = {"mu": dhhf_mu_real, "sigma": dhhf_sig, "fee": dhhf_fee}
else:
    params_ghhf = {"mu": ghhf_mu, "sigma": ghhf_sig, "fee": ghhf_fee}
    params_dhhf = {"mu": dhhf_mu, "sigma": dhhf_sig, "fee": dhhf_fee}

# Initialize session state for storing simulation results
if 'ghhf_final' not in st.session_state:
    st.session_state.ghhf_final = None
    st.session_state.dhhf_final = None
    st.session_state.simulation_success = False

# Run simulation only when button is clicked
if update_button:
    with st.spinner("Running simulations…"):
        ghhf_final, dhhf_final = run_dual_sim(
            params_ghhf, params_dhhf, years, sims, start_value, monthly_contrib, seed
        )
    
    # Store results in session state
    st.session_state.ghhf_final = ghhf_final
    st.session_state.dhhf_final = dhhf_final
    
    # Set success flag
    st.session_state.simulation_success = True
    
    # Rerun to show the success message
    st.rerun()
else:
    # Use stored results from session state
    ghhf_final = st.session_state.ghhf_final
    dhhf_final = st.session_state.dhhf_final

# Only show results if we have simulation data
if ghhf_final is not None and dhhf_final is not None:
    # -----------------------------
    # Results
    # -----------------------------
    if show_real_values:
        st.info(f"💰 **Real Values Displayed**: All results are inflation-adjusted using {inflation_rate*100:.1f}% annual inflation rate. Values shown represent purchasing power in today's dollars.")

    # Add KPI metrics for GHHF and DHHF
    st.subheader("Portfolio Statistics")
    
    # Calculate statistics
    ghhf_mean = np.mean(ghhf_final)
    ghhf_median = np.median(ghhf_final)
    ghhf_std = np.std(ghhf_final)
    
    dhhf_mean = np.mean(dhhf_final)
    dhhf_median = np.median(dhhf_final)
    dhhf_std = np.std(dhhf_final)
    
    # Display in three columns: combined means, combined medians, combined standard deviations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Mean (GHHF / DHHF)",
            f"{format_currency_compact(ghhf_mean)} / {format_currency_compact(dhhf_mean)}",
            help="Average terminal values: GHHF / DHHF"
        )
    
    with col2:
        st.metric(
            "Median (GHHF / DHHF)",
            f"{format_currency_compact(ghhf_median)} / {format_currency_compact(dhhf_median)}",
            help="Middle values when all outcomes are sorted: GHHF / DHHF"
        )
    
    with col3:
        st.metric(
            "Std Dev (GHHF / DHHF)",
            f"{format_currency_compact(ghhf_std)} / {format_currency_compact(dhhf_std)}",
            help="Standard deviation of terminal values: GHHF / DHHF"
        )
    
    # Add comparison metrics
    #st.subheader("Portfolio Comparison")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean_diff = ghhf_mean - dhhf_mean
        mean_diff_pct = (mean_diff / dhhf_mean) * 100
        st.metric(
            "Mean Difference (GHHF - DHHF)",
            format_currency_compact(mean_diff),
            f"{mean_diff_pct:+.1f}%",
            help="Difference in average outcomes between strategies"
        )
    
    with col2:
        median_diff = ghhf_median - dhhf_median
        median_diff_pct = (median_diff / dhhf_median) * 100
        st.metric(
            "Median Difference (GHHF - DHHF)",
            format_currency_compact(median_diff),
            f"{median_diff_pct:+.1f}%",
            help="Difference in median outcomes between strategies"
        )
    
    with col3:
        vol_ratio = ghhf_std / dhhf_std
        st.metric(
            "Volatility Ratio (GHHF/DHHF)",
            f"{vol_ratio:.2f}x",
            help="How much more volatile GHHF is compared to DHHF"
        )

    # Break-even percentile KPI
    #st.subheader("Break-even Analysis")

    # Find break-even percentile (where GHHF = DHHF)
    break_even_found = False
    break_even_pct = None

    # Check if GHHF ever catches up to DHHF
    if np.percentile(ghhf_final, 99) > np.percentile(dhhf_final, 99):
        # GHHF outperforms at high percentiles, find break-even
        for p in range(1, 100):
            if np.percentile(ghhf_final, p) >= np.percentile(dhhf_final, p):
                break_even_pct = p
                break_even_found = True
                break

    if break_even_found:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Break-even Percentile",
                f"{break_even_pct}th",
                help="GHHF starts outperforming DHHF from this percentile onwards"
            )
        with col2:
            ghhf_break_even = np.percentile(ghhf_final, break_even_pct)
            dhhf_break_even = np.percentile(dhhf_final, break_even_pct)
            st.metric(
                "Break-even Value",
                format_currency_compact(ghhf_break_even),
                help=f"Portfolio value at {break_even_pct}th percentile"
            )
        with col3:
            prob_outperform = 100 - break_even_pct
            st.metric(
                "Probability GHHF > DHHF",
                f"{prob_outperform}%",
                help=f"Chance that GHHF outperforms DHHF"
            )
    else:
        st.warning("GHHF does not outperform DHHF at any percentile in this simulation")

    # Ensure break-even percentile is included in the percentiles list if found
    if break_even_found and break_even_pct not in percentiles:
        percentiles.append(break_even_pct)
        percentiles.sort()  # Keep percentiles in ascending order

    #st.subheader("Percentile comparison (grouped bars)")
    # Grouped bars at selected percentiles
    vals_ghhf = [np.percentile(ghhf_final, p) for p in percentiles]
    vals_dhhf = [np.percentile(dhhf_final, p) for p in percentiles]

    # Calculate percentage differences for tooltips
    pct_diffs = [round(((ghhf_val - dhhf_val) / dhhf_val * 100), 1) for ghhf_val, dhhf_val in zip(vals_ghhf, vals_dhhf)]

    # Create Plotly bar chart
    fig2 = go.Figure()

    # Add GHHF bars
    fig2.add_trace(go.Bar(
        x=[f"{p}th" for p in percentiles],
        y=vals_ghhf,
        name="GHHF",
        marker_color='#2E86AB',
        hovertemplate="<b>GHHF</b><br>" +
                     "Percentile: %{x}<br>" +
                     "Value: %{customdata[0]}<br>" +
                     "vs DHHF: %{customdata[1]:+.0f}%<br>" +
                     "<extra></extra>",
        customdata=[[format_currency_compact(y), pct] for y, pct in zip(vals_ghhf, pct_diffs)]
    ))

    # Add DHHF bars
    fig2.add_trace(go.Bar(
        x=[f"{p}th" for p in percentiles],
        y=vals_dhhf,
        name="DHHF",
        marker_color='#A23B72',
        hovertemplate="<b>DHHF</b><br>" +
                     "Percentile: %{x}<br>" +
                     "Value: %{customdata[0]}<br>" +
                     "vs GHHF: %{customdata[1]:+.0f}%<br>" +
                     "<extra></extra>",
        customdata=[[format_currency_compact(y), -pct] for y, pct in zip(vals_dhhf, pct_diffs)]
    ))

    # Add ThinkCell-style percentage increase annotations
    for i, (p, ghhf_val, dhhf_val, pct_diff) in enumerate(zip(percentiles, vals_ghhf, vals_dhhf, pct_diffs)):
        # Calculate position for annotation (above the higher bar)
        max_val = max(ghhf_val, dhhf_val)
        y_pos = max_val + (max_val * 0.08)  # 8% above the higher bar for more spacing
        
        # Determine color based on whether GHHF outperforms DHHF
        if pct_diff > 0:
            color = '#2E86AB'  # GHHF blue when outperforming
            symbol = "▲"
        elif pct_diff < 0:
            color = '#A23B72'  # DHHF purple when outperforming
            symbol = "▼"
        else:
            color = '#666666'  # Gray when equal
            symbol = "●"
        
        # Add percentage annotation
        fig2.add_annotation(
            x=f"{p}th",
            y=y_pos,
            text=f"{symbol} {pct_diff:+.0f}%",
            showarrow=False,
            font=dict(
                size=16,
                color=color,
                family="Arial Black"
            ),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor=color,
            borderwidth=2,
            borderpad=6,
            align="center"
        )

    # Update layout
    fig2.update_layout(
        title="GHHF vs DHHF by Monte Carlo Percentile of Outcomes",
        xaxis_title="Percentile of outcomes",
        yaxis_title="Terminal value (A$)",
        barmode='group',
        height=800,
        showlegend=True,
        hovermode='closest'
    )

    # Remove gridlines and format y-axis with compact currency labels
    fig2.update_xaxes(showgrid=False)

    # Format y-axis with dynamic scaling based on data range
    max_val = max(max(vals_ghhf), max(vals_dhhf))
    min_val = min(min(vals_ghhf), min(vals_dhhf))
    data_range = max_val - min_val

    if max_val >= 1_000_000:
        # For millions, use dynamic increments based on data range
        if data_range >= 10_000_000:
            increment = 2_000_000  # 2M increments for very large ranges
        elif data_range >= 5_000_000:
            increment = 1_000_000  # 1M increments for large ranges
        else:
            increment = 500_000    # 500k increments for smaller ranges
        
        tickvals = list(np.arange(0, max_val + increment, increment))
        fig2.update_yaxes(
            showgrid=False,
            tickmode='array',
            tickvals=tickvals,
            ticktext=[f"${x/1_000_000:.1f}m" for x in tickvals]
        )
    elif max_val >= 100_000:
        # For hundreds of thousands, use dynamic increments
        if data_range >= 500_000:
            increment = 200_000  # 200k increments for large ranges
        elif data_range >= 200_000:
            increment = 100_000  # 100k increments for medium ranges
        else:
            increment = 50_000   # 50k increments for smaller ranges
        
        tickvals = list(np.arange(0, max_val + increment, increment))
        fig2.update_yaxes(
            showgrid=False,
            tickmode='array',
            tickvals=tickvals,
            ticktext=[f"${x/1_000:.0f}k" for x in tickvals]
        )
    elif max_val >= 1_000:
        # For thousands, use dynamic increments
        if data_range >= 50_000:
            increment = 20_000  # 20k increments for large ranges
        elif data_range >= 20_000:
            increment = 10_000  # 10k increments for medium ranges
        else:
            increment = 5_000   # 5k increments for smaller ranges
        
        tickvals = list(np.arange(0, max_val + increment, increment))
        fig2.update_yaxes(
            showgrid=False,
            tickmode='array',
            tickvals=tickvals,
            ticktext=[f"${x/1_000:.0f}k" for x in tickvals]
        )
    else:
        # For small values, use standard formatting
        fig2.update_yaxes(
            showgrid=False,
            tickformat=",.0f",
            tickprefix="$"
        )

    # Layout with chart on left, text on right
    col_chart, col_text = st.columns([2, 1])  # wider chart, narrower text

    with col_chart:
        st.plotly_chart(fig2, use_container_width=True)

    with col_text:
        st.markdown("### Description")
        st.markdown(
            "The chart presents terminal portfolio values from Monte Carlo simulations, "
            "shown across selected percentiles. Outcomes for the leveraged portfolio (GHHF, in **blue**) "
            "are displayed alongside the unleveraged portfolio (DHHF, in **purple**). Each bar represents "
            "the simulated portfolio value at a given percentile, with annotations showing the percentage "
            "difference between the two strategies. Upward arrows indicate scenarios where leverage outperforms, "
            "while downward arrows highlight underperformance."
        )

        st.markdown("### Insights")
        st.markdown(
            "When evaluating leveraged portfolios, the average outcome often appears highly attractive. "
            "However, averages can be misleading because they are disproportionately influenced by the best-performing "
            "scenarios. A more robust assessment comes from examining the distribution of results across percentiles. "
            "This reveals that, while leverage amplifies gains in favourable markets, it also magnifies losses and "
            "increases exposure to *sequence of returns risk*. In practice, there can be extended periods—sometimes "
            "spanning 20 years or more—where a leveraged portfolio underperforms its unleveraged counterpart. "
            "The graph highlights this trade-off: leverage offers the potential for higher upside, but also carries a "
            "significant risk of prolonged underperformance, depending on the timing and path of returns."
        )

        st.markdown("### Notes")
        st.markdown(
            "These results are based on Monte Carlo simulations, which generate outcomes using randomised return "
            "paths rather than historical sequences. Unlike bootstrap or block-bootstrap approaches, Monte Carlo "
            "simulations do not assume time-series features such as mean reversion. This makes them more conservative "
            "at the extreme ends of the distribution, particularly when assessing tail risks and the likelihood of "
            "extreme outcomes."
        )

    # Add month-to-month pathway chart
    st.subheader("Portfolio Evolution Over Time")
    
    # Add toggle for view type
    col_toggle, col_percentile = st.columns([1, 1])
    with col_toggle:
        view_type = st.radio(
            "Chart View:",
            ["GHHF all percentiles", "GHHF vs DHHF Comparison"],
            horizontal=True
        )
    
    with col_percentile:
        if view_type == "GHHF vs DHHF Comparison":
            selected_percentile = st.selectbox(
                "Select Percentile to Compare:",
                percentiles,
                format_func=lambda x: f"{x}th percentile"
            )
        else:
            selected_percentile = None
    
    # Function to calculate month-to-month pathways for each decile
    def calculate_pathways_for_percentiles(percentiles_list, years, sims, start_value, monthly_contrib, seed, include_dhhf=False):
        """Calculate month-to-month pathways for specific percentiles"""
        months = years * 12
        
        # Use the same random seed for consistency
        rng = np.random.default_rng(seed)
        shocks = rng.normal(0.0, 1.0, size=(sims, months))
        
        def simulate_pathway_with_shocks(mu_annual, sigma_annual, fee_annual):
            drift = (mu_annual - 0.5 * sigma_annual**2) / 12.0
            vol = sigma_annual / np.sqrt(12.0)
            gross_returns = np.exp(drift + vol * shocks)
            fee_factor = (1.0 - fee_annual / 12.0)
            
            # Calculate wealth at each month for all simulations
            wealth_paths = np.zeros((sims, months))
            wealth = np.full(sims, start_value, dtype=float)
            
            for m in range(months):
                wealth += monthly_contrib
                wealth *= gross_returns[:, m]
                wealth *= fee_factor
                wealth_paths[:, m] = wealth
            
            return wealth_paths
        
        # Get pathways for GHHF
        ghhf_paths = simulate_pathway_with_shocks(params_ghhf["mu"], params_ghhf["sigma"], params_ghhf["fee"])
        
        # Get pathways for DHHF if comparison is requested
        dhhf_paths = None
        if include_dhhf:
            dhhf_paths = simulate_pathway_with_shocks(params_dhhf["mu"], params_dhhf["sigma"], params_dhhf["fee"])
        
        # Calculate average pathway for each percentile
        pathways = {}
        for p in percentiles_list:
            # Find the value at this percentile for each month
            ghhf_percentile_path = []
            dhhf_percentile_path = []
            
            for m in range(months):
                ghhf_percentile_path.append(np.percentile(ghhf_paths[:, m], p))
                if include_dhhf:
                    dhhf_percentile_path.append(np.percentile(dhhf_paths[:, m], p))
            
            pathways[f"{p}th"] = {
                'ghhf': ghhf_percentile_path
            }
            if include_dhhf:
                pathways[f"{p}th"]['dhhf'] = dhhf_percentile_path
        
        return pathways, months
    
    # Calculate pathways based on view type
    if view_type == "GHHF vs DHHF Comparison":
        # For comparison view, only calculate the selected percentile
        pathways, months = calculate_pathways_for_percentiles([selected_percentile], years, sims, start_value, monthly_contrib, seed, include_dhhf=True)
    else:
        # For GHHF-only view, calculate all percentiles
        pathways, months = calculate_pathways_for_percentiles(percentiles, years, sims, start_value, monthly_contrib, seed, include_dhhf=False)
    
    # Focus on first 5 years for better visibility of sequence risk
    focus_months = min(60, months)  # Show first 5 years (60 months) or all if less
    month_labels = [f"Month 0"] + [f"Year {m//12 + 1}, Month {m%12 + 1}" for m in range(focus_months)]
    
    # Create the pathway chart
    fig_pathways = go.Figure()
    
    # Add lines for each percentile based on view type
    if view_type == "GHHF vs DHHF Comparison":
        # Comparison view: show GHHF vs DHHF for selected percentile
        percentile_label = f"{selected_percentile}th"
        pathway_data = pathways[percentile_label]
        
        # Add GHHF line (solid)
        ghhf_values = [start_value] + pathway_data['ghhf'][:focus_months]
        fig_pathways.add_trace(go.Scatter(
            x=month_labels,
            y=ghhf_values,
            mode='lines',
            name=f"GHHF {percentile_label}",
            line=dict(color='#2E86AB', width=3),
            hovertemplate=f"<b>GHHF {percentile_label}</b><br>" +
                         "Month: %{x}<br>" +
                         "Value: %{y:,.0f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add DHHF line (dashed)
        dhhf_values = [start_value] + pathway_data['dhhf'][:focus_months]
        fig_pathways.add_trace(go.Scatter(
            x=month_labels,
            y=dhhf_values,
            mode='lines',
            name=f"DHHF {percentile_label}",
            line=dict(color='#A23B72', width=3, dash='dash'),
            hovertemplate=f"<b>DHHF {percentile_label}</b><br>" +
                         "Month: %{x}<br>" +
                         "Value: %{y:,.0f}<br>" +
                         "<extra></extra>"
        ))
        
        chart_title = f"GHHF vs DHHF Comparison: {percentile_label} Percentile - Sequence of Returns Risk"
        
    else:
        # GHHF-only view: show all percentiles
        colors_ghhf = ['#2E86AB', '#5DA9E9', '#8BC34A', '#FFC107', '#FF9800', '#F44336', '#9C27B0', '#673AB7', '#3F51B5']
        
        for i, (percentile_label, pathway_data) in enumerate(pathways.items()):
            color_idx = i % len(colors_ghhf)
            
            # Add GHHF line for focused time period, starting with month 0
            ghhf_values = [start_value] + pathway_data['ghhf'][:focus_months]
            
            fig_pathways.add_trace(go.Scatter(
                x=month_labels,
                y=ghhf_values,
                mode='lines',
                name=f"GHHF {percentile_label}",
                line=dict(color=colors_ghhf[color_idx], width=2),
                hovertemplate=f"<b>GHHF {percentile_label}</b><br>" +
                             "Month: %{x}<br>" +
                             "Value: %{y:,.0f}<br>" +
                             "<extra></extra>"
            ))
        
        chart_title = "GHHF Portfolio Evolution: Early Years Focus - Sequence of Returns Risk"
    
    # Update layout
    fig_pathways.update_layout(
        title=chart_title,
        xaxis_title="Time Period (First 5 Years)",
        yaxis_title="Portfolio Value (A$) - Log Scale",
        height=600,
        showlegend=True,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotations to highlight sequence of returns risk (only for GHHF-only view)
    if view_type != "GHHF vs DHHF Comparison" and focus_months >= 12:
        # Find the 1st and 50th percentile values at year 1
        year1_1st = pathways["1th"]['ghhf'][11] if "1th" in pathways else None
        year1_50th = pathways["50th"]['ghhf'][11] if "50th" in pathways else None
        
        if year1_1st and year1_50th:
            # Add annotation showing the gap at year 1
            fig_pathways.add_annotation(
                x="Year 1, Month 12",
                y=year1_50th,
                text=f"Year 1 Gap: {((year1_50th - year1_1st) / year1_1st * 100):.0f}%",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                font=dict(size=12, color="red"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=1
            )
    
    # Format x-axis to show monthly labels for first year, then yearly
    if focus_months >= 12:
        # Show monthly labels for first year, then yearly
        tickvals = []
        ticktext = []
        
        # Month 0
        tickvals.append("Month 0")
        ticktext.append("Month 0")
        
        # First year: monthly
        for m in range(12):
            tickvals.append(month_labels[m+1])  # +1 because month 0 is now at index 0
            ticktext.append(f"Month {m+1}")
        
        # Subsequent years: yearly
        for m in range(12, focus_months, 12):
            tickvals.append(month_labels[m+1])  # +1 because month 0 is now at index 0
            ticktext.append(f"Year {m//12 + 1}")
        
        fig_pathways.update_xaxes(
            showgrid=False,
            tickangle=45,
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext
        )
    else:
        # If less than 1 year, show all months including month 0
        fig_pathways.update_xaxes(
            showgrid=False,
            tickangle=45
        )
    
    # Set y-axis to log scale but with better range for early years
    fig_pathways.update_yaxes(
        type="log",
        showgrid=False,
        tickformat=",.0f",
        tickprefix="$"
    )
    
    # Add a horizontal line at the starting investment value
    fig_pathways.add_hline(
        y=start_value,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        opacity=0.5
    )
    
    # Display the pathway chart
    # Layout with chart on left, text on right
    col_chart, col_text = st.columns([2, 1])  # wider chart, narrower text
    
    with col_chart:
        st.plotly_chart(fig_pathways, use_container_width=True)
    
    with col_text:
        st.markdown("### Chart Explanation")
        if view_type == "GHHF vs DHHF Comparison":
            st.markdown(
                f"This chart compares the portfolio evolution of GHHF (leveraged) and DHHF (unleveraged) strategies "
                f"at the {selected_percentile}th percentile of outcomes. Each line represents the average portfolio "
                f"value at each month across 20,000 simulations that end up at this specific percentile."
            )
        else:
            st.markdown(
                "This chart shows how GHHF portfolio values evolve over time for different percentiles of outcomes. "
                "Each line represents the average portfolio value at each month across 20,000 simulations that end up "
                "at a specific percentile."
            )
        
        st.markdown("### Key Results")
        st.markdown(
            "**Sequence of Returns Risk**: Early declines in portfolio value create gaps that cannot be recovered "
            "in leveraged portfolios. The log scale reveals how small early setbacks compound into large long-term "
            "differences, demonstrating why poor first-year performance is so detrimental to leveraged strategies."
        )
        
        st.markdown("### Why This Matters")
        st.markdown(
            "In leveraged portfolios, poor early performance not only reduces the base investment amount but also "
            "diminishes the leverage multiplier effect. This creates a compounding penalty that makes it increasingly "
            "difficult to catch up over time, even with subsequent strong performance."
        )

    # Key percentiles table (duplicated)
    st.subheader("Investment Terminal Values")
    data = {
        "Percentile": [f"{p}th" for p in percentiles],
        "GHHF": [np.percentile(ghhf_final, p) for p in percentiles],
        "DHHF": [np.percentile(dhhf_final, p) for p in percentiles],
    }
    df = pd.DataFrame(data)

    # Calculate percentage differences
    df["Delta"] = df["GHHF"] - df["DHHF"]
    df["Delta %"] = ((df["GHHF"] - df["DHHF"]) / df["DHHF"] * 100).round(1)

    # Create display version with formatted currency
    df_display = df.copy()
    df_display["GHHF"] = df_display["GHHF"].map(aud)
    df_display["DHHF"] = df_display["DHHF"].map(aud)
    df_display["Delta"] = df_display["Delta"].map(aud)
    df_display["Delta %"] = df_display["Delta %"].map(lambda x: f"{x}%")

    # Apply color styling to the dataframe
    def color_rows(row):
        """Color entire rows based on whether GHHF outperforms DHHF, with break-even highlighted in blue"""
        # Get the Delta % value from the row
        delta_pct_str = row['Delta %']
        percentile_str = row['Percentile']
        
        # Check if this is the break-even percentile
        if break_even_found and percentile_str == f"{break_even_pct}th":
            return ['background-color: #cce5ff; color: #004085'] * len(row)  # Blue for break-even
        
        # Apply performance-based coloring
        if isinstance(delta_pct_str, str) and delta_pct_str.endswith('%'):
            try:
                # Extract numeric value from percentage string
                num_val = float(delta_pct_str.rstrip('%'))
                if num_val > 0:
                    return ['background-color: #d4edda; color: #155724'] * len(row)  # Green for positive
                elif num_val < 0:
                    return ['background-color: #f8d7da; color: #721c24'] * len(row)  # Red for negative
                else:
                    return ['background-color: #e2e3e5; color: #383d41'] * len(row)  # Gray for zero
            except:
                return [''] * len(row)
        return [''] * len(row)

    # Apply styling to entire rows
    styled_df = df_display.style.apply(color_rows, axis=1)

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Notes
    st.info(
        """
        **Notes**
        - This is a stylised Monte Carlo using geometric Brownian motion with monthly steps.
        - Fees are applied as a constant drag each month (MER/12).
        - Assumptions are *yours to set* — this tool doesn't fetch live data. Consider stress-testing a range of returns/volatility/fees.
        - Using the same random shocks for both series ensures a fair, apples-to-apples comparison under identical market paths.
        """
    )
else:
    # Show message when no simulation has been run yet
    pass
