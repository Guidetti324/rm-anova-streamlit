
# rm_anova_streamlit_app.py
# -----------------------------------------------------------------------------
# Repeated-Measures ANOVA – Complete Hand-Calculation Walk-Through (Streamlit)
# Rewritten from the original React/TSX component to a Streamlit app.
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------- PAGE SETUP ----------------------------------
st.set_page_config(
    page_title="Repeated-Measures ANOVA – Hand Calculations",
    layout="wide",
    page_icon="📊",
)

st.title("Repeated-Measures ANOVA – Complete Hand-Calculation Walk-Through")
st.write(
    "Use the button below to generate a fresh random data set. "
    "All workings update automatically so you can follow the mechanics with new numbers each time."
)


# ------------------------------- CONSTANTS -----------------------------------
CONDITION_LABELS: List[str] = ["Pre (Week 0)", "Week 4", "Week 8"]
K_CONDITIONS: int = len(CONDITION_LABELS)
DEFAULT_N_PARTICIPANTS: int = 12


# ------------------------------ HELPER FUNCS ---------------------------------
def rnd(x: float, dp: int = 4) -> float:
    \"\"\"Round to dp decimal places; safe for non-finite inputs.\"\"\"
    if x is None or not np.isfinite(x):
        return float("nan")
    f = 10 ** dp
    return math.floor(x * f + 0.5) / f


def generate_data(n: int = DEFAULT_N_PARTICIPANTS, seed: int | None = None) -> np.ndarray:
    \"\"\"Generate a pseudo-random repeated-measures data grid of shape [n, 3].
    Baseline (Pre) is 9–18 inclusive; then small declines to Week 4 and Week 8.
    \"\"\"
    rng = np.random.default_rng(seed)
    pre = rng.integers(9, 19, size=n)  # 9–18 inclusive
    wk4 = np.maximum(5, pre - rng.integers(0, 5, size=n))
    wk8 = np.maximum(3, wk4 - rng.integers(0, 5, size=n))
    data = np.column_stack([pre, wk4, wk8]).astype(float)
    return data


def transpose(matrix: np.ndarray) -> np.ndarray:
    return matrix.T


@dataclass
class RMStats:
    condition_means: np.ndarray
    grand_mean: float
    participant_means: np.ndarray

    ss_treatment: float
    df_treatment: int

    ss_subjects: float
    df_subjects: int

    ss_total: float
    df_total: int

    ss_error: float
    df_error: int

    ms_treatment: float
    ms_error: float
    F: float

    eta2: float
    eta2p: float


def compute_statistics(data: np.ndarray) -> RMStats:
    n, k = data.shape

    # Step 1 – condition means & grand mean
    condition_means = data.mean(axis=0)
    grand_mean = float(data.mean())

    # Step 3 – participant means
    participant_means = data.mean(axis=1)

    # Step 2 – SS_Treatment and df_Treatment
    ss_treatment = float(n * np.sum((condition_means - grand_mean) ** 2))
    df_treatment = k - 1

    # Step 4 – SS_Subjects and df_Subjects
    ss_subjects = float(k * np.sum((participant_means - grand_mean) ** 2))
    df_subjects = n - 1

    # Step 5 – SS_Total and df_Total
    ss_total = float(np.sum((data - grand_mean) ** 2))
    df_total = n * k - 1

    # Step 6 – SS_Error and df_Error
    ss_error = float(ss_total - ss_treatment - ss_subjects)
    df_error = df_treatment * df_subjects

    # Step 7 – Mean Squares and F
    ms_treatment = float(ss_treatment / df_treatment)
    ms_error = float(ss_error / df_error)
    F = float(ms_treatment / ms_error)

    # Step 8 – Effect sizes
    eta2 = float(ss_treatment / ss_total) if ss_total > 0 else float("nan")
    eta2p = float(ss_treatment / (ss_treatment + ss_error)) if (ss_treatment + ss_error) > 0 else float("nan")

    return RMStats(
        condition_means=condition_means,
        grand_mean=grand_mean,
        participant_means=participant_means,
        ss_treatment=ss_treatment,
        df_treatment=df_treatment,
        ss_subjects=ss_subjects,
        df_subjects=df_subjects,
        ss_total=ss_total,
        df_total=df_total,
        ss_error=ss_error,
        df_error=df_error,
        ms_treatment=ms_treatment,
        ms_error=ms_error,
        F=F,
        eta2=eta2,
        eta2p=eta2p,
    )


def f_critical(alpha: float, dfn: int, dfd: int) -> float:
    \"\"\"Return F critical value at the upper tail for given alpha.
    Tries SciPy; falls back to a small lookup for (2, 22) used here.
    \"\"\"
    try:
        from scipy.stats import f  # type: ignore
        return float(f.ppf(1 - alpha, dfn, dfd))
    except Exception:
        if (dfn, dfd) == (2, 22):
            return 3.4434  # F_{0.95,(2,22)} ≈ 3.4434
        return float("nan")


def paired_t_table(data: np.ndarray, absolute: bool = False) -> Tuple[pd.DataFrame, List[Dict[str, float]]]:
    \"\"\"Compute pairwise paired-samples summary (Diff, Diff^2, sums, sd, t) for
    the three comparisons: Pre–Week4, Week4–Week8, Pre–Week8.
    If absolute=True, uses |diff| to match the original artefact; otherwise uses signed differences (recommended).
    Returns the long table and a list of per-comparison summaries.
    \"\"\"
    n = data.shape[0]
    comparisons = [("Pre (Week 0)", 0, "Week 4", 1), ("Week 4", 1, "Week 8", 2), ("Pre (Week 0)", 0, "Week 8", 2)]
    long_rows = []
    summaries = []
    for (lab_a, ia, lab_b, ib) in comparisons:
        d = data[:, ia] - data[:, ib]
        if absolute:
            d = np.abs(d)
        d2 = d ** 2

        sum_d = float(d.sum())
        sum_d2 = float(d2.sum())
        mean_d = sum_d / n
        # Unbiased SD of differences
        var_d = (sum_d2 - (sum_d ** 2) / n) / (n - 1) if n > 1 else float("nan")
        var_d = max(var_d, 0.0)
        sd_d = math.sqrt(var_d)
        t_stat = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else (math.inf if mean_d != 0 else 0.0)

        summaries.append(
            dict(
                label=f"{lab_a} – {lab_b}",
                sum_diff=sum_d,
                sum_diff2=sum_d2,
                mean=mean_d,
                sd=sd_d,
                t=t_stat,
            )
        )

        for i in range(n):
            long_rows.append(
                {
                    "Participant": f"P{i+1}",
                    f"Diff ({lab_a} - {lab_b})": float(d[i]),
                    "Diff²": float(d2[i]),
                    "Comparison": f"{lab_a} & {lab_b}",
                }
            )

        # Totals row (to be rendered separately per comparison)
    long_df = pd.DataFrame(long_rows)
    return long_df, summaries


# ---------------------------- SIDEBAR CONTROLS -------------------------------
with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Random seed (optional)", min_value=0, value=0, step=1)
    use_seed = st.checkbox("Use seed", value=False, help="Tick to make the random data reproducible.")
    alpha = st.selectbox("Significance level (α) for F-test", options=[0.10, 0.05, 0.01], index=1)
    use_signed = st.checkbox(
        "Use signed differences for paired t-tests (recommended)",
        value=True,
        help="Untick to replicate the original artefact's use of absolute differences."
    )
    st.caption("Design is fixed at n = 12 participants and k = 3 time-points to mirror the original artefact.")


# ----------------------------- SESSION STATE ---------------------------------
if "data" not in st.session_state:
    st.session_state["data"] = generate_data(DEFAULT_N_PARTICIPANTS, seed if use_seed else None)

if st.button("Generate New Data"):
    st.session_state["data"] = generate_data(DEFAULT_N_PARTICIPANTS, seed if use_seed else None)

data = np.array(st.session_state["data"], dtype=float)
n, k = data.shape
assert k == 3, "This app expects exactly three time-points."

# Offer dataset download
df_data = pd.DataFrame(data.astype(int), columns=CONDITION_LABELS, index=[f"P{i+1}" for i in range(n)])
st.download_button(
    label="Download dataset as CSV",
    data=df_data.to_csv().encode("utf-8"),
    file_name="rm_anova_dataset.csv",
    mime="text/csv",
)


# -------------------------------- STEP 0 -------------------------------------
st.subheader("Step 0: Raw Data Table")
st.dataframe(df_data, use_container_width=True)


# -------------------------------- STEP 1 -------------------------------------
st.subheader("Step 1: Calculate every group mean and the grand mean")
stats = compute_statistics(data)

# Display the per-condition means with explicit fraction-style LaTeX
def latex_mean_of_column(nums: List[int], n: int, label: str, mean_val: float) -> None:
    # Join numbers as a + b + c in LaTeX
    num_str = " + ".join(str(int(x)) for x in nums)
    st.latex(rf"\bar{{X}}_{{{label}}} = \frac{{{num_str}}}{{{n}}} = {mean_val:.6f}")


col_values = [data[:, j].astype(int).tolist() for j in range(k)]
latex_mean_of_column(col_values[0], n, "Pre", stats.condition_means[0])
latex_mean_of_column(col_values[1], n, "Week\ 4", stats.condition_means[1])
latex_mean_of_column(col_values[2], n, "Week\ 8", stats.condition_means[2])

flat_nums = " + ".join(str(int(x)) for x in data.ravel())
st.latex(rf"\text{{Grand Mean}} = \frac{{{flat_nums}}}{{{n*k}}} = {stats.grand_mean:.6f}")


# -------------------------------- STEP 2 -------------------------------------
st.subheader("Step 2: Calculate SS₍Treatment₎ and df₍Treatment₎")
dev2 = (stats.condition_means - stats.grand_mean) ** 2
df_step2 = pd.DataFrame(
    {
        "Condition": CONDITION_LABELS,
        "Mean": [rnd(x, 6) for x in stats.condition_means],
        "(X̄_j − X̄_Grand)²": [rnd(x, 6) for x in dev2],
    }
)
st.table(df_step2)
st.markdown(
    f"**SS**$_\\text{{Treatment}}$ = n · Σ[(X̄$_j$ − X̄$_\\text{{Grand}}$)²] "
    f"= {n} × {rnd(np.sum(dev2), 6)} = **{rnd(stats.ss_treatment, 6)}**"
)
st.markdown(
    f"**df**$_\\text{{Treatment}}$ = k − 1 = {k} − 1 = **{stats.df_treatment}**"
)


# -------------------------------- STEP 3 -------------------------------------
st.subheader("Step 3: Calculate participant means")
df_step3 = df_data.copy()
df_step3["Participant Mean"] = [rnd(x, 6) for x in stats.participant_means]
st.table(df_step3)


# -------------------------------- STEP 4 -------------------------------------
st.subheader("Step 4: Calculate SS₍Subjects₎ and df₍Subjects₎")
terms = [(m - stats.grand_mean) ** 2 for m in stats.participant_means]
inside_sum = " + ".join(f"({rnd(stats.participant_means[i],2)} − {rnd(stats.grand_mean,2)})^2" for i in range(n))
st.latex(rf"SS_{{Subjects}} = k \sum ( \bar{{X}}_i - \bar{{X}}_{{Grand}} )^2 = {k} \times \left[ {inside_sum} \right] = {stats.ss_subjects:.6f}")
st.markdown(f"**df**$_\\text{{Subjects}}$ = n − 1 = {n} − 1 = **{stats.df_subjects}**")


# -------------------------------- STEP 5 -------------------------------------
st.subheader("Step 5: Calculate SS₍Total₎ and df₍Total₎")
rows_5 = []
for i in range(n):
    for j in range(k):
        xij = data[i, j]
        term = (xij - stats.grand_mean) ** 2
        rows_5.append(
            {
                "Participant": f"P{i+1}",
                "Time Point": CONDITION_LABELS[j],
                "Score": int(xij),
                "(X_ij − X̄_Grand)²": rnd(term, 6),
            }
        )
df_step5 = pd.DataFrame(rows_5)
with st.expander("Show working table for SS_Total", expanded=False):
    st.dataframe(df_step5, use_container_width=True, height=400)
st.markdown(
    f"**SS**$_\\text{{Total}}$ = Σ[(X$_{{ij}}$ − X̄$_\\text{{Grand}}$)²] = **{rnd(stats.ss_total, 6)}**"
)
st.markdown(
    f"**df**$_\\text{{Total}}$ = kn − 1 = {k}×{n} − 1 = **{stats.df_total}**"
)


# -------------------------------- STEP 6 -------------------------------------
st.subheader("Step 6: Calculate SS₍Error₎ and df₍Error₎")
st.markdown(
    f"**SS**$_\\text{{Error}}$ = SS$_\\text{{Total}}$ − SS$_\\text{{Treatment}}$ − SS$_\\text{{Subjects}}$ "
    f"= {rnd(stats.ss_total,6)} − {rnd(stats.ss_treatment,6)} − {rnd(stats.ss_subjects,6)} = **{rnd(stats.ss_error,6)}**"
)
st.markdown(
    f"**df**$_\\text{{Error}}$ = (k − 1)(n − 1) = {stats.df_treatment} × {stats.df_subjects} = **{stats.df_error}**"
)


# -------------------------------- STEP 7 -------------------------------------
st.subheader("Step 7: Calculate MS₍Treatment₎, MS₍Error₎ and the F-statistic")
st.latex(r"MS_{\text{Treatment}} = \frac{SS_{\text{Treatment}}}{df_{\text{Treatment}}}")
st.markdown(
    f"= {rnd(stats.ss_treatment,6)} / {stats.df_treatment} = **{rnd(stats.ms_treatment,6)}**"
)
st.latex(r"MS_{\text{Error}} = \frac{SS_{\text{Error}}}{df_{\text{Error}}}")
st.markdown(
    f"= {rnd(stats.ss_error,6)} / {stats.df_error} = **{rnd(stats.ms_error,6)}**"
)
st.latex(r"F = \frac{MS_{\text{Treatment}}}{MS_{\text{Error}}}")
st.markdown(f"= {rnd(stats.ms_treatment,6)} / {rnd(stats.ms_error,6)} = **{rnd(stats.F,6)}**")


# -------------------------------- STEP 8 -------------------------------------
st.subheader("Step 8: Calculate η² and η²ₚ")
st.latex(r"\eta^2 = \frac{SS_{\text{Treatment}}}{SS_{\text{Total}}}")
st.markdown(f"= {rnd(stats.ss_treatment,6)} / {rnd(stats.ss_total,6)} = **{rnd(stats.eta2,6)}**")
st.latex(r"\eta^2_p = \frac{SS_{\text{Treatment}}}{SS_{\text{Treatment}} + SS_{\text{Error}}}")
st.markdown(
    f"= {rnd(stats.ss_treatment,6)} / ({rnd(stats.ss_treatment,6)} + {rnd(stats.ss_error,6)}) = **{rnd(stats.eta2p,6)}**"
)


# ---------------------------- STEPS 9–12 (t tests) ---------------------------
st.subheader("Step 9: Determine statistical significance of F and decide on H₀")
crit = f_critical(alpha=float(alpha), dfn=stats.df_treatment, dfd=stats.df_error)
st.markdown(
    f"Calculated statistic: **F**$\\_\\text{{calculated}}$({stats.df_treatment}, {stats.df_error}) = {rnd(stats.F,3)}"
)
if np.isfinite(crit):
    st.markdown(f"Using α = {alpha} and df = ({stats.df_treatment}, {stats.df_error}), the critical value is ≈ **{rnd(crit, 2)}**.")
    decision = "reject" if stats.F > crit else "fail to reject"
    st.markdown(f"Because F$\\_\\text{{calculated}}$ { 'exceeds' if stats.F > crit else 'does not exceed' } F$\\_\\text{{critical}}$, we **{decision}** H₀.")
else:
    st.markdown("F critical value unavailable (SciPy not installed). For df = (2,22) and α = 0.05, use ≈ **3.44**.")


st.subheader("Step 10: List Bonferroni-corrected pairwise comparisons")
m = K_CONDITIONS * (K_CONDITIONS - 1) // 2
st.markdown(f"With {K_CONDITIONS} conditions there are **{m}** pairwise comparisons:")
st.markdown("1. Pre (Week 0) vs Week 4  \n2. Week 4 vs Week 8  \n3. Pre (Week 0) vs Week 8")

st.subheader("Step 11: Compute differences, means and standard deviations")
long_df, summaries = paired_t_table(data, absolute=not use_signed)

# Render three small tables side-by-side (Diff and Diff²)
cols = st.columns(3)
pairs = [("Pre (Week 0)", "Week 4"), ("Week 4", "Week 8"), ("Pre (Week 0)", "Week 8")]
for idx, (lab_a, lab_b) in enumerate(pairs):
    with cols[idx]:
        part = long_df[long_df["Comparison"] == f"{lab_a} & {lab_b}"].copy()
        totals_row = pd.DataFrame(
            {
                "Participant": ["Σ"],
                f"Diff ({lab_a} - {lab_b})": [rnd(part[f"Diff ({lab_a} - {lab_b})"].sum(), 4)],
                "Diff²": [rnd(part["Diff²"].sum(), 4)],
                "Comparison": [f"{lab_a} & {lab_b}"],
            }
        )
        display_df = pd.concat([part[["Participant", f"Diff ({lab_a} - {lab_b})", "Diff²"]], totals_row[["Participant", f"Diff ({lab_a} - {lab_b})", "Diff²"]]], ignore_index=True)
        st.table(display_df)

st.subheader("Step 12: Calculate the t-statistic for each comparison")
st.latex(r"t = \frac{\text{Mean of Differences}}{\text{SD of Differences} / \sqrt{n}}")
for s in summaries:
    st.markdown(
        f"**t**$_{{{s['label']}}}$ = {rnd(s['mean'], 2)} / ({rnd(s['sd'], 4)} / √{n}) = **{rnd(s['t'], 3)}**"
    )


st.caption("© 2025 Astra for Dr Oliver Guidetti – All calculations rendered in real time.")
