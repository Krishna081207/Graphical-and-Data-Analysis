import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import streamlit as st
import openpyxl
# Page config (must be set before any Streamlit calls)
st.set_page_config(page_title="Graphical Analysis + ML Models", layout="wide")

st.info("This is an enhanced graphical analysis and ML model dashboard. Upload your data and explore various visualizations and ML models.")
st.info("Ensure your dataset has sufficient numeric columns and datasets for analysis and modeling.")
#UI ux for page
st.markdown("""
<style>
    /* Main app background - Green to Blue gradient */
    .stApp {
        background: linear-gradient(135deg, #00aa00 0%, #0000ff 100%);
        background-attachment: fixed;
    }

    /* Main container */
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Info box */
    .stInfo {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-left: 4px solid #000000 !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Typography - Black Font - ALL BOLD */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 900;
        letter-spacing: 0.5px;
    }

    h1 {
        border-bottom: 3px solid rgba(0,0,0,0.15);
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-size: 2.5em;
    }

    h2 {
        border-bottom: 2px solid rgba(0,0,0,0.12);
        padding-bottom: 8px;
        font-size: 2em;
    }

    h3 {
        font-size: 1.5em;
    }

    /* Text content */
    body, p, label, span, div, .stMarkdown, .stText {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #000000 !important;
        line-height: 1.6;
        font-weight: 700;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #000000;
        background-color: rgba(0, 0, 0, 0.05) !important;
        font-weight: 700;
        color: #000000 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00aa00 0%, #0000ff 100%);
        color: #000000 !important;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 900;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #0000ff 0%, #00aa00 100%);
        box-shadow: 0 4px 12px rgba(0, 170, 0, 0.4);
        transform: translateY(-2px);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    textarea {
        border: 2px solid rgba(0,0,0,0.2) !important;
        border-radius: 6px !important;
        padding: 10px !important;
        font-family: 'Segoe UI', sans-serif !important;
        color: #000000 !important;
        font-weight: 700 !important;
        background-color: rgba(0,0,0,0.05) !important;
        transition: border-color 0.3s ease;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    textarea:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.06) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #00aa00 0%, #0000ff 100%);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .css-1v3fvcr {
        font-weight: 700 !important;
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stNumberInput > div > div > input,
    [data-testid="stSidebar"] .stSelectbox > div > div > select {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 2px solid rgba(0, 0, 0, 0.06) !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Metrics */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(0, 0, 0, 0.5);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .stMetric > div > div > div {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    /* Checkbox and radio */
    .stCheckbox > label,
    .stRadio > label {
        font-weight: 900;
        color: #000000 !important;
    }

    /* Success/Error/Warning messages */
    .stSuccess, .stError, .stWarning {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stSuccess { background-color: rgba(0,170,68,0.12) !important; border-left: 4px solid #00aa44 !important; }
    .stError   { background-color: rgba(255,0,0,0.12) !important; border-left: 4px solid #ff0000 !important; }
    .stWarning { background-color: rgba(255,165,0,0.12) !important; border-left: 4px solid #ff6600 !important; }

    /* Dataframe styling */
    .stDataFrame, .stTable {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #000000 !important;
    }
    .stDataFrame table td, .stDataFrame table th {
        color: #000000 !important;
    }

    /* Divider */
    hr {
        border: 0;
        border-top: 2px solid rgba(0,0,0,0.06);
        margin: 20px 0;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00aa00 0%, #0000ff 100%);
    }

    /* Links and small text */
    a, small {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

            
# Title input
graph_title = st.text_input("Enter title of graph", value="Graphical Analysis & ML Models")
st.title(graph_title)

# File uploader
data = st.file_uploader("Upload your data", type=['csv', 'xlsx'])

if data is None:
    st.info("Please upload a CSV or Excel file to get started.")
    st.stop()

# Load data
try:
    if data.name.endswith('.csv'):
        df = pd.read_csv(data)
    else:
        df = pd.read_excel(data)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# --- NEW: normalize student/name column and set RowID index starting at 1 ---
student_col = next((c for c in df.columns if c.lower() in ('student', 'name')), None)
if student_col:
    # keep original name column, add a uniform StudentName column (string)
    df['StudentName'] = df[student_col].astype(str)
else:
    # ensure column exists so UI logic is simpler
    df['StudentName'] = None

# add RowID column (1-based) and set as index so index shows from 1
df.insert(0, 'RowID', range(1, len(df) + 1))
df.set_index('RowID', inplace=True)

# sidebar option: use student names as x-axis tick labels when plotting
use_student_labels = False
if student_col:
    use_student_labels = st.sidebar.checkbox("Use student names as x-axis labels", value=True)

st.write(f"**Original Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ========== DATA FILTERING SECTION ==========
st.sidebar.header("🔍 Data Selection")

# Select specific columns
st.sidebar.subheader("Column Selection")
all_columns = list(df.columns)
selected_columns = st.sidebar.multiselect("Select columns to analyze", all_columns, default=all_columns)

if not selected_columns:
    st.error("Please select at least one column.")
    st.stop()

df_filtered = df[selected_columns].copy()

# Select specific rows by range
st.sidebar.subheader("Row Selection")
row_range_option = st.sidebar.radio("Select rows by:", ["All rows", "Row range", "First N rows"])

if row_range_option == "Row range":
    max_rows = len(df_filtered)
    start_row = st.sidebar.number_input("Start row (0-indexed)", min_value=0, max_value=max_rows-1, value=0)
    end_row = st.sidebar.number_input("End row (0-indexed, inclusive)", min_value=start_row, max_value=max_rows-1, value=min(start_row+10, max_rows-1))
    df_filtered = df_filtered.iloc[start_row:end_row+1].copy()
    st.write(f"✅ Selected rows {start_row} to {end_row}")

elif row_range_option == "First N rows":
    n_rows = st.sidebar.number_input("Number of rows to display", min_value=1, max_value=len(df_filtered), value=10)
    df_filtered = df_filtered.head(n_rows).copy()
    st.write(f"✅ Selected first {n_rows} rows")

else:
    st.write(f"✅ Using all {len(df_filtered)} rows")

# Filter by column conditions
st.sidebar.subheader("Column Filtering (Optional)")
if st.sidebar.checkbox("Apply column-based filters"):
    filter_col = st.sidebar.selectbox("Select column to filter", selected_columns)
    numeric_cols_temp = df_filtered.select_dtypes(include=np.number).columns.tolist()
    
    if filter_col in numeric_cols_temp:
        filter_type = st.sidebar.radio("Filter type", ["Greater than", "Less than", "Between", "Equal to"])
        
        if filter_type == "Greater than":
            threshold = st.sidebar.number_input(f"{filter_col} > ", value=0.0)
            df_filtered = df_filtered[df_filtered[filter_col] > threshold]
            st.write(f"✅ Filtered: {filter_col} > {threshold}")
        
        elif filter_type == "Less than":
            threshold = st.sidebar.number_input(f"{filter_col} < ", value=100.0)
            df_filtered = df_filtered[df_filtered[filter_col] < threshold]
            st.write(f"✅ Filtered: {filter_col} < {threshold}")
        
        elif filter_type == "Between":
            min_val = st.sidebar.number_input(f"{filter_col} min", value=0.0)
            max_val = st.sidebar.number_input(f"{filter_col} max", value=100.0)
            df_filtered = df_filtered[(df_filtered[filter_col] >= min_val) & (df_filtered[filter_col] <= max_val)]
            st.write(f"✅ Filtered: {min_val} ≤ {filter_col} ≤ {max_val}")
        
        elif filter_type == "Equal to":
            val = st.sidebar.number_input(f"{filter_col} = ", value=0.0)
            df_filtered = df_filtered[df_filtered[filter_col] == val]
            st.write(f"✅ Filtered: {filter_col} == {val}")
    else:
        st.sidebar.info(f"{filter_col} is not numeric. Cannot filter.")

st.write(f"**Filtered Dataset Shape:** {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns")

st.write("**Data Preview:**")
st.dataframe(df_filtered.head(10))

st.write("**Statistical Summary:**")
st.write(df_filtered.describe())

# Numeric columns
numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()

# --- X-axis label helpers & UI ---
string_cols_all = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
label_column = st.sidebar.selectbox(
    "Select column for x-axis labels",
    ["Index (1, 2, 3...)", "None"] + string_cols_all,
    index=0
)
label_rotation = st.sidebar.slider("X-axis label rotation", 0, 90, 45)

def get_x_labels(df_data, label_col):
    """Get labels for x-axis based on selected column"""
    if label_col in (None, "None"):
        return [str(i) for i in df_data.index]
    if label_col == "Index (1, 2, 3...)":
        return [str(i) for i in df_data.index]
    if label_col in df_data.columns:
        return df_data[label_col].astype(str).values.tolist()
    return [str(i) for i in df_data.index]

def apply_custom_labels(ax, df_data, label_col, rotation=45):
    """Apply custom x-axis labels to the plot (auto-subsample if too many)"""
    labels = get_x_labels(df_data, label_col)
    n = len(labels)
    if n == 0:
        return
    # For many labels, show only a subset to avoid overlap
    max_labels = 40
    if n > max_labels:
        step = max(1, n // max_labels)
        ticks = list(range(0, n, step))
        tick_labels = [labels[i] for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=rotation, ha='right')
    else:
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=rotation, ha='right')
    ax.tick_params(axis='x', which='major', labelsize=8)

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns for correlation analysis.")
else:
    # Correlation heatmap
    st.write("**Correlation Matrix:**")
    corr = df_filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    st.pyplot(fig)
    plt.close()

# ========== ENHANCED VISUALIZATION SECTION ==========
st.write("---")
st.subheader("📊 Data Visualization")
graph_type = st.selectbox(
    "Select graph type",
    ["Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Bar Graph",
     "Violin Plot", "KDE Plot", "Count Plot", "Multiple Series Bar Chart", "Grouped Bar Chart"]
)

all_cols = list(df_filtered.columns)
numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

# select primary column (any)
selected_col = st.selectbox("Select primary column", all_cols)

# Additional options for bar/grouped graphs
bar_agg = None
groupby_col = None
if "Bar" in graph_type or graph_type == "Grouped Bar Chart":
    st.subheader(f"🎯 {graph_type} Options")
    bar_agg = st.selectbox("Aggregation method", ["Sum", "Mean", "Count", "Max", "Min", "Std Dev"])
    if st.checkbox("Group by another column"):
        group_cols = [c for c in df_filtered.columns if c != selected_col]
        if group_cols:
            groupby_col = st.selectbox("Group by column", group_cols)
        else:
            groupby_col = None

fig, ax = plt.subplots(figsize=(12, 6))

def plot_value_counts(col, axis):
    vc = df_filtered[col].value_counts()
    colors = plt.cm.viridis(np.linspace(0, 1, len(vc)))
    vc.plot(kind='bar', ax=axis, color=colors)
    axis.set_xlabel(col)
    axis.set_ylabel("Count")
    axis.set_title(f'Value Counts of {col}')
    plt.xticks(rotation=label_rotation, ha='right')

if graph_type == "Histogram":
    if selected_col in numeric_cols:
        sns.histplot(df_filtered[selected_col], kde=True, ax=ax, color='skyblue', bins=30)
        ax.set_title(f'Histogram of {selected_col}')
        ax.set_xlabel(selected_col); ax.set_ylabel("Frequency")
    else:
        # fallback: show value counts for categorical/string
        plot_value_counts(selected_col, ax)

elif graph_type == "Box Plot":
    if selected_col in numeric_cols:
        sns.boxplot(y=df_filtered[selected_col], ax=ax, color='lightgreen')
        ax.set_title(f'Box Plot of {selected_col}'); ax.set_ylabel(selected_col)
    else:
        st.warning("Box plot requires a numeric column.")

elif graph_type == "Violin Plot":
    if selected_col in numeric_cols:
        sns.violinplot(y=df_filtered[selected_col], ax=ax, color='lightcoral')
        ax.set_title(f'Violin Plot of {selected_col}'); ax.set_ylabel(selected_col)
    else:
        st.warning("Violin plot requires a numeric column.")

elif graph_type == "KDE Plot":
    if selected_col in numeric_cols:
        df_filtered[selected_col].plot(kind='density', ax=ax, color='blue', linewidth=2)
        ax.set_title(f'KDE Plot of {selected_col}'); ax.set_xlabel(selected_col); ax.set_ylabel("Density")
    else:
        st.warning("KDE requires a numeric column.")

elif graph_type == "Line Plot":
    # For strings show value_counts line; for numeric plot values (per-row)
    if selected_col in numeric_cols:
        ax.plot(df_filtered[selected_col].values, marker='o', linestyle='-', linewidth=2, color='blue')
        ax.set_ylabel(selected_col)
    else:
        # plot counts to show trend across categories
        vc = df_filtered[selected_col].value_counts()
        ax.plot(vc.values, marker='o', linestyle='-', linewidth=2, color='blue')
        ax.set_ylabel("Count")
        # set x labels to categories
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels([str(x) for x in vc.index], rotation=label_rotation, ha='right')
    ax.set_title(f'Line Plot of {selected_col}')
    ax.set_xlabel("Index")
    ax.grid(True, alpha=0.3)
    apply_custom_labels(ax, df_filtered, label_column, rotation=label_rotation)

elif graph_type == "Bar Graph":
    # if column numeric but many unique values -> histogram recommended; otherwise show counts
    if selected_col in cat_cols:
        plot_value_counts(selected_col, ax)
    elif selected_col in numeric_cols and df_filtered[selected_col].nunique() <= 50:
        plot_value_counts(selected_col, ax)
    elif selected_col in numeric_cols:
        sns.histplot(df_filtered[selected_col], kde=False, ax=ax, color='coral', bins=30)
        ax.set_title(f'Histogram of {selected_col}'); ax.set_xlabel(selected_col); ax.set_ylabel("Frequency")
    else:
        st.warning("Cannot create bar graph for the selected column.")

elif graph_type == "Scatter Plot":
    # allow categorical -> encode to numeric for plotting; x-axis can be chosen among columns
    if len(all_cols) >= 2:
        other_col = st.selectbox("Select second column for scatter plot", [c for c in all_cols if c != selected_col])
        x = df_filtered[selected_col]
        y = df_filtered[other_col]
        # encode object columns
        if x.dtype == object or x.dtype.name == 'category':
            x_plot = x.astype('category').cat.codes
            x_label_map = x.astype('category').cat.categories
        else:
            x_plot = x
            x_label_map = None
        if y.dtype == object or y.dtype.name == 'category':
            y_plot = y.astype('category').cat.codes
            y_label_map = y.astype('category').cat.categories
        else:
            y_plot = y
            y_label_map = None
        scatter = ax.scatter(x_plot, y_plot, alpha=0.7, c='purple', s=80)
        ax.set_xlabel(selected_col); ax.set_ylabel(other_col)
        ax.set_title(f'Scatter Plot: {selected_col} vs {other_col}')
        if x_label_map is not None and len(x_label_map) <= 50:
            ax.set_xticks(range(len(x_label_map)))
            ax.set_xticklabels([str(u) for u in x_label_map], rotation=label_rotation, ha='right')
        if y_label_map is not None and len(y_label_map) <= 50:
            ax.set_yticks(range(len(y_label_map)))
            ax.set_yticklabels([str(u) for u in y_label_map], rotation=0)
        ax.grid(True, alpha=0.3)
    else:
        st.warning("Need at least two columns for scatter plot.")

elif graph_type == "Count Plot":
    if cat_cols:
        cat_col = st.selectbox("Select categorical column", cat_cols)
        plot_value_counts(cat_col, ax)
        ax.set_title(f'Count Plot of {cat_col}')
    else:
        st.warning("No categorical columns available.")

elif graph_type == "Multiple Series Bar Chart":
    if len(numeric_cols) >= 2:
        cols_to_plot = st.multiselect("Select numeric columns to plot", numeric_cols,
                                     default=numeric_cols[:min(3, len(numeric_cols))])
        if cols_to_plot:
            df_filtered[cols_to_plot].plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Multiple Series Bar Chart'); ax.set_xlabel("Index"); ax.set_ylabel("Values")
            ax.legend(title="Columns"); plt.xticks(rotation=label_rotation)
            apply_custom_labels(ax, df_filtered, label_column, rotation=label_rotation)
    else:
        st.warning("Need at least 2 numeric columns.")

elif graph_type == "Grouped Bar Chart":
    if cat_cols and len(numeric_cols) >= 1:
        cat_col = st.selectbox("Select categorical column for grouping", cat_cols)
        numeric_col_for_group = st.selectbox("Select numeric column for values", numeric_cols)
        try:
            grouped_data = df_filtered.groupby(cat_col)[numeric_col_for_group].agg(bar_agg.lower())
        except Exception:
            grouped_data = df_filtered.groupby(cat_col)[numeric_col_for_group].mean()
        grouped_data.plot(kind='bar', ax=ax, color='salmon')
        ax.set_title(f'Grouped Bar Chart ({bar_agg} of {numeric_col_for_group} by {cat_col})')
        ax.set_xlabel(cat_col); ax.set_ylabel(f"{bar_agg} of {numeric_col_for_group}"); plt.xticks(rotation=label_rotation, ha='right')
    else:
        st.warning("Need at least one categorical and one numeric column.")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Display numeric statistics
st.subheader(f"📈 Numeric Statistics for {selected_col}")

if selected_col in numeric_cols:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{df_filtered[selected_col].mean():.2f}")
    col2.metric("Median", f"{df_filtered[selected_col].median():.2f}")
    col3.metric("Std Dev", f"{df_filtered[selected_col].std():.2f}")
    col4.metric("Count", f"{df_filtered[selected_col].count()}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min", f"{df_filtered[selected_col].min():.2f}")
    col6.metric("Max", f"{df_filtered[selected_col].max():.2f}")
    col7.metric("Q1 (25%)", f"{df_filtered[selected_col].quantile(0.25):.2f}")
    col8.metric("Q3 (75%)", f"{df_filtered[selected_col].quantile(0.75):.2f}")
else:
    st.info("Selected column is not numeric — numeric summary not available.")

# ========== ML MODELS SECTION ==========
st.sidebar.header("🤖 ML Models Analysis")

if st.sidebar.checkbox("Run ML Models"):
    st.subheader("Machine Learning Models")
    
    all_cols = list(df_filtered.columns)
    target = st.sidebar.selectbox("Select target column (what to predict)", all_cols)
    
    features = st.sidebar.multiselect("Select feature columns", [c for c in all_cols if c != target])
    
    if not features:
        features = [c for c in all_cols if c != target and c in numeric_cols]
    
    if target not in df_filtered.columns:
        st.error("Invalid target column.")
        st.stop()
    
    if not features:
        st.error("Select at least one feature column.")
        st.stop()

    # Prepare data
    try:
        X = df_filtered[features].copy()
        y = df_filtered[target].copy()
        
        # Drop NaN rows
        data_clean = pd.concat([X, y], axis=1).dropna()
        X = data_clean[features]
        y = data_clean[target]
        
        # Validate marks (0-100 constraint)
        if 'marks' in target.lower() or 'attendance' in target.lower():
            y = y.clip(0, 100)
            st.info(f"✅ Marks/Attendance clamped to range [0, 100]")
        
        if len(X) < 10:
            st.error("Not enough data samples (need at least 10).")
            st.stop()
        
        st.write(f"📊 Training samples: {len(X)}, Features: {X.shape[1]}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "K-Nearest Neighbors (KNN)", "Random Forest"])
        
        # Train models
        if model_choice == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_name = "Linear Regression"
            
        elif model_choice == "K-Nearest Neighbors (KNN)":
            k = st.sidebar.slider("Select K value", 1, 20, 5)
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_name = f"KNN (K={k})"
            
        else:  # Random Forest
            n_trees = st.sidebar.slider("Number of trees", 10, 200, 100)
            model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_name = f"Random Forest ({n_trees} trees)"
        
        # Clamp predictions to 0-100 for marks/attendance
        if 'marks' in target.lower() or 'attendance' in target.lower():
            y_pred = np.clip(y_pred, 0, 100)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        st.subheader(f"📈 {model_name} Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{r2:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("MSE", f"{mse:.4f}")
        
        # Visualization: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions', s=100)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs Predicted ({model_name})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Residuals plot
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(y_pred, residuals, alpha=0.6, color='green', s=100)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # ========== AUTOMATED COMMENTS SECTION ==========
        st.subheader("📝 Model Analysis & Comments")
        
        def generate_comment(r2_val, mae_val, rmse_val, model_type):
            comment = f"**Model:** {model_type}\n\n"
            
            if r2_val > 0.9:
                comment += "✅ **Excellent Fit:** R² > 0.9 — Model explains 90%+ variance. Very reliable predictions.\n\n"
            elif r2_val > 0.7:
                comment += "✅ **Good Fit:** R² > 0.7 — Model explains 70%+ variance. Reasonably accurate predictions.\n\n"
            elif r2_val > 0.5:
                comment += "⚠️ **Moderate Fit:** R² > 0.5 — Model explains 50%+ variance. Predictions have moderate accuracy.\n\n"
            else:
                comment += "❌ **Poor Fit:** R² ≤ 0.5 — Model has low explanatory power. Consider adding features or using different model.\n\n"
            
            comment += f"**Mean Absolute Error (MAE):** {mae_val:.2f} — On average, predictions are off by {mae_val:.2f} units.\n\n"
            comment += f"**Root Mean Squared Error (RMSE):** {rmse_val:.2f} — Penalizes larger errors; predictions ≈ ±{rmse_val:.2f} units.\n\n"
            
            if model_type == "Linear Regression" and r2_val < 0.6:
                comment += "💡 **Suggestion:** Linear Regression assumes linear relationships. Consider KNN or Random Forest for non-linear patterns.\n\n"
            elif model_type.startswith("KNN") and r2_val < 0.6:
                comment += "💡 **Suggestion:** Try increasing K value or use Random Forest for better generalization.\n\n"
            elif model_type.startswith("Random Forest") and r2_val < 0.6:
                comment += "💡 **Suggestion:** Increase number of trees or add more relevant features.\n\n"
            
            if 'marks' in target.lower() or 'attendance' in target.lower():
                comment += f"✏️ **Note:** Predictions are clamped to [0, 100] range for marks/attendance.\n\n"
                avg_error = mae_val
                if avg_error < 5:
                    comment += "✅ **Marks Prediction Accuracy:** Excellent — Average error < 5 points.\n\n"
                elif avg_error < 10:
                    comment += "✅ **Marks Prediction Accuracy:** Good — Average error < 10 points.\n\n"
                else:
                    comment += "⚠️ **Marks Prediction Accuracy:** Moderate — Average error >= 10 points.\n\n"
            
            return comment
        
        comment_text = generate_comment(r2, mae, rmse, model_name)
        st.markdown(comment_text)
        
        # Feature importances (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            st.subheader("🎯 Feature Importances")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax, palette='viridis')
            ax.set_title("Feature Importances")
            st.pyplot(fig)
            plt.close()
        
        st.success("✅ ML Analysis Complete!")
        
    except Exception as e:
        st.error(f"❌ Error in ML analysis: {e}")

