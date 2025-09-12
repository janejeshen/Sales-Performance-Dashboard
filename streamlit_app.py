# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ===================== PAGE SETUP =====================
st.set_page_config(page_title="SALES PERFORMANCE DASHBOARD", layout="wide")
st.markdown(
    """
    <style>
      .big-title {font-size:48px; font-weight:800; color:#0a69ff; text-align:center; margin:6px 0 18px;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="big-title">SALES PERFORMANCE DASHBOARD</div>', unsafe_allow_html=True)

# ===================== HELPERS =====================
ALT_NAMES = {
    "Order Date":      ["order date", "order_date", "date"],
    "Order ID":        ["order id", "order-id", "order number", "order_no", "order no", "orderid"],
    "Category":        ["category", "product category"],
    "Sub-Category":    ["sub-category", "product sub-category", "subcategory", "sub category"],
    "Segment":         ["segment", "customer segment", "market segment"],
    "Order Priority":  ["order priority", "priority"],
    "Customer Name":   ["customer name", "customer", "client name", "client"],
    "Product Name":    ["product name", "product"],
    "Sales":           ["sales", "revenue", "amount", "total"],
    "Profit":          ["profit", "margin", "gross profit"],
    "Quantity":        ["quantity", "qty", "units"]
}

MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

def _stripna(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaT": np.nan})

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename_map = {}
    for target, alts in ALT_NAMES.items():
        if target in df.columns:
            continue
        for alt in alts:
            if alt in cols_lower:
                rename_map[cols_lower[alt]] = target
                break
    return df.rename(columns=rename_map) if rename_map else df

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = alias_columns(df)

    # Ensure required basics exist
    for c in ["Order Date", "Sales"]:
        if c not in df.columns:
            df[c] = np.nan

    # Parse dates
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if "Ship Date" in df.columns:
        df["Ship Date"] = pd.to_datetime(df.get("Ship Date"), errors="coerce")

    # Ensure other columns exist
    optional = ["Order ID","Category","Sub-Category","Segment","Order Priority",
                "Customer Name","Product Name","Profit","Quantity"]
    for c in optional:
        if c not in df.columns:
            df[c] = np.nan

    # Clean strings (fix slicers)
    for c in ["Category","Sub-Category","Segment","Order Priority","Customer Name","Product Name"]:
        df[c] = _stripna(df[c])

    # Force numeric
    df = _to_numeric(df, ["Sales","Profit","Quantity"])

    # Time helpers
    df["Year"] = df["Order Date"].dt.year
    df["QuarterNum"] = df["Order Date"].dt.quarter
    df["Quarter"] = "Q" + df["QuarterNum"].astype("Int64").astype(str)
    df["MonthNum"] = df["Order Date"].dt.month
    df["MonthName"] = pd.Categorical(
        df["Order Date"].dt.month_name(),
        categories=MONTH_NAMES, ordered=True
    )
    return df

@st.cache_data(show_spinner=False)
def load_local():
    # Priority: CSV → XLSX → XLS
    csv_p = Path("superstore.csv")
    xlsx_p = Path("Sample - Superstore Sales.xlsx")
    xls_p  = Path("Sample - Superstore Sales.xls")
    if csv_p.exists():
        return normalize_df(pd.read_csv(csv_p))
    if xlsx_p.exists():
        return normalize_df(pd.read_excel(xlsx_p, engine="openpyxl"))
    if xls_p.exists():
        return normalize_df(pd.read_excel(xls_p, engine="xlrd"))  # needs xlrd
    return None

def opts(series: pd.Series, stringy=False):
    s = series.dropna()
    if stringy:
        s = _stripna(s).dropna()
        values = sorted(s.unique().tolist())
    else:
        try:
            values = sorted(s.unique().tolist())
        except Exception:
            values = sorted(map(str, s.unique().tolist()))
    return ["All"] + values

# ===================== LOAD DATA =====================
df_local = load_local()
with st.sidebar:
    uploaded = st.file_uploader("Upload Superstore CSV/XLSX/XLS (optional)", type=["csv","xlsx","xls"])

if uploaded is not None:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = normalize_df(pd.read_csv(uploaded))
    elif name.endswith(".xlsx"):
        df = normalize_df(pd.read_excel(uploaded, engine="openpyxl"))
    else:
        df = normalize_df(pd.read_excel(uploaded, engine="xlrd"))
else:
    if df_local is None:
        st.warning("Add `superstore.csv` or `Sample - Superstore Sales.xls/xlsx` next to this script, or upload via the sidebar.")
        st.stop()
    df = df_local

# ===================== FILTERS (TOP BAR) =====================
years      = opts(df["Year"])
quarters   = ["All","Q1","Q2","Q3","Q4"]
subcats    = opts(df["Sub-Category"], stringy=True)   # works with Product Sub-Category (aliased)
priorities = opts(df["Order Priority"], stringy=True)

c1, c2, c3, c4, c5 = st.columns([1,1,1,1,0.3])
with c1: sel_year     = st.selectbox("Year", years, index=0, key="f_year")
with c2: sel_quarter  = st.selectbox("Quarter", quarters, index=0, key="f_quarter")
with c3: sel_subcat   = st.selectbox("Product Sub-Category", subcats, index=0, key="f_subcat")
with c4: sel_priority = st.selectbox("Order Priority", priorities, index=0, key="f_priority")
with c5:
    if st.button("↻ Reset", use_container_width=True):
        st.session_state["f_year"] = "All"
        st.session_state["f_quarter"] = "All"
        st.session_state["f_subcat"] = "All"
        st.session_state["f_priority"] = "All"
        st.rerun()

# Apply filters
dff = df.copy()
if sel_year != "All":
    dff = dff[dff["Year"] == sel_year]
if sel_quarter != "All":
    dff = dff[dff["QuarterNum"] == int(sel_quarter[1])]
if sel_subcat != "All":
    dff = dff[dff["Sub-Category"] == sel_subcat]
if sel_priority != "All":
    dff = dff[dff["Order Priority"] == sel_priority]

# Safety: clean again after filtering
for c in ["Category","Sub-Category","Segment"]:
    dff[c] = _stripna(dff[c])

if dff.empty or dff["Order Date"].isna().all():
    st.info("No data matches the current filters.")
    st.stop()

# ===================== ROW 1 =====================
r1c1, r1c2, r1c3 = st.columns(3)

# --- Monthly Orders (bar)
with r1c1:
    orders = (
        dff.dropna(subset=["Order Date"])
           .groupby(["MonthName","MonthNum"], as_index=False)["Order ID"]
           .nunique()
           .sort_values("MonthNum")
           .rename(columns={"Order ID":"Orders"})
    )
    fig = px.bar(
        orders, x="MonthName", y="Orders",
        title="Monthly Orders", text="Orders",
        category_orders={"MonthName": MONTH_NAMES},
    )
    fig.update_layout(yaxis_title=None, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# --- Sales by Segment & Category (GROUPED BAR)
with r1c2:
    seg_cat = (
        dff.dropna(subset=["Segment","Category"])
           .groupby(["Segment","Category"], as_index=False)["Sales"].sum()
    )
    if len(seg_cat):
        pref = ["Corporate","Home Office","Consumer","Small Business"]
        order = [s for s in pref if s in seg_cat["Segment"].unique()]
        if order:
            seg_cat["Segment"] = pd.Categorical(seg_cat["Segment"], categories=order, ordered=True)
            seg_cat = seg_cat.sort_values("Segment")

        fig2 = px.bar(
            seg_cat,
            x="Segment", y="Sales", color="Category",
            barmode="group", text_auto=".2s",
            title="Sales by Segment & Category"
        )
        fig2.update_yaxes(tickformat="~s")
        fig2.update_layout(xaxis_title=None, yaxis_title=None, legend_title="Product Category")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No Segment/Category data under current filters.")

# --- Monthly Sales Trend (YoY / Aggregate / MA) — SAFE REINDEX (no dtype mismatch)
with r1c3:
    view = st.radio(
        "Trend view",
        ["By year (YoY)", "Aggregate", "Aggregate + 3-mo MA"],
        horizontal=True, key="trend_view",
    )

    # Base monthly totals by Year x Month
    ts = (
        dff.dropna(subset=["Order Date"])
           .groupby(["Year", "MonthNum"], as_index=False)["Sales"]
           .sum()
    )

    # Fill missing months with NaN using MultiIndex (avoids the groupby.apply dtype issue)
    years_present = pd.Index(sorted(ts["Year"].dropna().unique().tolist()), name="Year")
    months_all = pd.Index(range(1, 13), name="MonthNum")
    full_index = pd.MultiIndex.from_product([years_present, months_all], names=["Year","MonthNum"])
    ts = (
        ts.set_index(["Year","MonthNum"])
          .reindex(full_index)
          .reset_index()
    )

    # Month names in correct order
    ts["MonthName"] = pd.Categorical(
        [MONTH_NAMES[int(m-1)] if pd.notna(m) else None for m in ts["MonthNum"]],
        categories=MONTH_NAMES, ordered=True
    )

    if view == "By year (YoY)":
        fig3 = px.line(
            ts, x="MonthName", y="Sales", color="Year",
            title="Monthly Sales Trend (YoY)", markers=True
        )
        fig3.update_traces(line=dict(width=3))  # crisp line
        fig3.update_yaxes(tickformat="~s")
        fig3.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig3, use_container_width=True)

    elif view == "Aggregate":
        agg = (
            ts.groupby("MonthName", as_index=False)["Sales"]
              .sum()
              .sort_values("MonthName")
        )
        fig3 = px.line(
            agg, x="MonthName", y="Sales",
            title="Monthly Sales Trend", markers=True
        )
        fig3.update_traces(line=dict(width=3))
        fig3.update_yaxes(tickformat="~s")
        fig3.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig3, use_container_width=True)

    else:  # "Aggregate + 3-mo MA"
        agg = (
            ts.groupby("MonthNum", as_index=False)["Sales"].sum()
              .sort_values("MonthNum")
        )
        agg["MonthName"] = pd.Categorical(
            [MONTH_NAMES[int(m-1)] for m in agg["MonthNum"]],
            categories=MONTH_NAMES, ordered=True
        )
        agg["MA3"] = agg["Sales"].rolling(3, min_periods=1).mean()

        fig3 = px.line(
            agg, x="MonthName", y=["Sales", "MA3"],
            title="Monthly Sales Trend (with 3-month MA)", markers=True
        )
        for tr in fig3.data:
            if tr.name == "MA3":
                tr.update(line=dict(width=4))
        fig3.update_yaxes(tickformat="~s")
        fig3.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None)
        st.plotly_chart(fig3, use_container_width=True)

# ===================== ROW 2 =====================
r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    top_cust_sales = (
        dff.dropna(subset=["Customer Name"])
           .groupby("Customer Name", as_index=False)["Sales"].sum()
           .nlargest(10, "Sales").sort_values("Sales")
    )
    fig4 = px.bar(top_cust_sales, x="Sales", y="Customer Name",
                  orientation="h", title="Top 10 Customers by Sales", text_auto=".2s")
    fig4.update_xaxes(tickformat="~s")
    fig4.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig4, use_container_width=True)

with r2c2:
    top_cust_profit = (
        dff.dropna(subset=["Customer Name"])
           .groupby("Customer Name", as_index=False)["Profit"].sum()
           .nlargest(10, "Profit").sort_values("Profit")
    )
    fig5 = px.bar(top_cust_profit, x="Profit", y="Customer Name",
                  orientation="h", title="Top 10 Customers by Profit", text_auto=".2s")
    fig5.update_xaxes(tickformat="~s")
    fig5.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig5, use_container_width=True)

with r2c3:
    top_prod_profit = (
        dff.dropna(subset=["Product Name"])
           .groupby("Product Name", as_index=False)["Profit"].sum()
           .nlargest(10, "Profit").sort_values("Profit")
    )
    fig6 = px.bar(top_prod_profit, x="Profit", y="Product Name",
                  orientation="h", title="Top 10 Products by Profit", text_auto=".2s")
    fig6.update_xaxes(tickformat="~s")
    fig6.update_layout(xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig6, use_container_width=True)

# ===================== DIAGNOSTICS (OPTIONAL) =====================
with st.sidebar.expander("🔎 Diagnostics", expanded=False):
    st.write("Columns:", list(df.columns))
    st.write("Segments:", sorted([x for x in df["Segment"].dropna().unique()])[:20])
    st.write("Categories:", sorted([x for x in df["Category"].dropna().unique()])[:20])
    st.write("Sub-Categories:", sorted([x for x in df["Sub-Category"].dropna().unique()])[:20])

st.caption("Built with Streamlit + Plotly • Superstore dataset")
