import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="UrbanPulse — Growth Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card2: #1a2235;
    --accent-cyan: #00e5ff;
    --accent-lime: #b8ff57;
    --accent-amber: #ffb347;
    --accent-red: #ff4f6b;
    --text-primary: #e8edf5;
    --text-muted: #6b7fa3;
    --border: #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #050810 0%, #0a0e1a 40%, #0d1525 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c16 0%, #0d1525 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Space Mono', monospace !important; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1f3c 0%, #071423 50%, #060f20 100%);
    border: 1px solid #1a2d4a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,229,255,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}
.hero-title span { color: var(--accent-cyan); }
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.6rem;
    margin-bottom: 0;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--card-accent, var(--accent-cyan));
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
    color: var(--text-primary);
    line-height: 1;
}
.metric-delta {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent-lime);
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text-primary);
    letter-spacing: 0.02em;
    padding: 0.4rem 0 0.8rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header .tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    background: rgba(0,229,255,0.12);
    color: var(--accent-cyan);
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* Zone cards */
.zone-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.zone-name { font-weight: 700; font-size: 0.9rem; color: var(--text-primary); }
.zone-meta { font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--text-muted); margin-top: 0.2rem; }
.zone-score {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
}
.score-hot { color: var(--accent-lime); }
.score-warm { color: var(--accent-amber); }
.score-cool { color: var(--accent-cyan); }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, rgba(0,229,255,0.05), rgba(184,255,87,0.04));
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent-cyan);
    line-height: 1.7;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), #0096c7) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,229,255,0.35) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
    background: transparent !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.15) !important;
    color: var(--accent-cyan) !important;
}

/* Sliders + inputs */
.stSlider [data-testid="stThumbValue"] { color: var(--accent-cyan) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Dataframe */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_engineer(path="train.csv"):
    df = pd.read_csv(path)


    cols = [
        'SalePrice', 'GrLivArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'GarageCars',
        'FullBath', 'BedroomAbvGr', 'Neighborhood', 'BldgType',
        'LotArea', 'MoSold', 'YrSold', 'SaleCondition',
        'Fireplaces', 'WoodDeckSF', 'OpenPorchSF'
    ]
    df = df[cols].dropna().copy()
    df.rename(columns={'SalePrice': 'price'}, inplace=True)

    df['price_per_sqft']   = df['price'] / df['GrLivArea']
    df['rent_estimate']    = df['price'] * 0.006
    df['rental_yield']     = (df['rent_estimate'] * 12) / df['price']

    df['age']              = 2024 - df['YearBuilt']
    df['years_since_reno'] = 2024 - df['YearRemodAdd']
    df['reno_ratio']       = 1 - (df['years_since_reno'] / df['age'].clip(lower=1))

    df['infra_score'] = (
        df['OverallQual'] * 0.4 +
        df['GarageCars']  * 0.8 +
        df['FullBath']    * 0.9 +
        df['Fireplaces']  * 0.3 +
        (df['TotalBsmtSF'] / df['TotalBsmtSF'].max()) * 3
    ).clip(1, 10)

    neigh_counts = df.groupby('Neighborhood')['price'].transform('count')
    df['listing_density'] = (neigh_counts / neigh_counts.max() * 10).clip(0, 10)

   
    neigh_median = df.groupby('Neighborhood')['price'].transform('median')
    df['pricing_velocity'] = ((df['price'] - neigh_median) / neigh_median).clip(-0.5, 1.0)

    df['underval_score'] = (
        (neigh_median - df['price']) / neigh_median
    ).clip(0, None)

   
    scaler = MinMaxScaler()
    features = ['pricing_velocity', 'rental_yield', 'infra_score',
                 'reno_ratio', 'listing_density', 'underval_score']
    scaled = scaler.fit_transform(df[features].fillna(0))
    weights = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    df['growth_velocity_score'] = (scaled @ weights) * 10  # 0-10 scale

  
    def classify(s):
        if s >= 7:   return "🔥 Hot Zone"
        elif s >= 5: return "🌤 Warm Zone"
        elif s >= 3: return "🌊 Stable Zone"
        else:        return "❄️ Cool Zone"
    df['zone_class'] = df['growth_velocity_score'].apply(classify)

    
    df['proj_24m'] = df['price'] * (1 + df['growth_velocity_score'] * 0.012)
    df['proj_60m'] = df['price'] * (1 + df['growth_velocity_score'] * 0.030)

    return df

@st.cache_resource
def train_model(df):
    features = ['price', 'infra_score', 'rental_yield',
                 'listing_density', 'pricing_velocity',
                 'reno_ratio', 'age', 'GrLivArea']
    X = df[features].fillna(0)
    y = df['growth_velocity_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, features, r2_score(y_test, preds), mean_absolute_error(y_test, preds)

df = load_and_engineer()
model, model_features, r2, mae = train_model(df)

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.15em;
                text-transform:uppercase; color:#6b7fa3; padding: 0.5rem 0 1rem;">
    ◈ UrbanPulse v2.0<br>Growth Intelligence Engine
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🗂 Data Filters**")
    neighborhoods = sorted(df['Neighborhood'].unique())
    selected_neigh = st.multiselect(
        "Neighborhoods", neighborhoods,
        default=neighborhoods[:8],
        help="Filter analysis by neighborhood"
    )

    price_min, price_max = int(df['price'].min()), int(df['price'].max())
    price_range = st.slider(
        "Price Range ($)", price_min, price_max,
        (price_min, min(price_min + 300000, price_max)),
        step=5000
    )

    zone_filter = st.multiselect(
        "Zone Classes",
        options=df['zone_class'].unique().tolist(),
        default=df['zone_class'].unique().tolist()
    )

    st.divider()
    st.markdown("**🤖 Model**")
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.68rem; color:#6b7fa3; line-height:1.8;">
    Algorithm: Gradient Boosting<br>
    R² Score: <span style="color:#b8ff57;">{r2:.3f}</span><br>
    MAE: <span style="color:#00e5ff;">{mae:.3f}</span><br>
    Train set: 80% | Test: 20%
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#3a4d6a; line-height:1.6;">
    DATA SOURCES<br>
    ▸ Ames Housing Dataset<br>
    ▸ Infrastructure proxies<br>
    ▸ Rental yield modeling<br>
    ▸ Growth velocity scoring
    </div>
    """, unsafe_allow_html=True)

filtered = df[
    df['Neighborhood'].isin(selected_neigh) &
    df['price'].between(*price_range) &
    df['zone_class'].isin(zone_filter)
].copy()

st.markdown("""
<div class="hero-banner">
    <p class="hero-sub">◈ Predictive Geospatial Analytics Engine</p>
    <h1 class="hero-title">Urban<span>Pulse</span> — Growth Intelligence</h1>
    <p style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#4a6080; margin:0.8rem 0 0;">
    Real estate hotspot detection · Rental yield analysis · 24–60 month horizon forecasting
    </p>
</div>
""", unsafe_allow_html=True)

hot_pct = (filtered['zone_class'] == "🔥 Hot Zone").mean() * 100
avg_gvs = filtered['growth_velocity_score'].mean()
avg_yield = filtered['rental_yield'].mean() * 100
avg_ppsf = filtered['price_per_sqft'].mean()
underval_count = (filtered['underval_score'] > 0.05).sum()

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, "#00e5ff", "AVG GROWTH SCORE", f"{avg_gvs:.1f}/10", f"▲ {hot_pct:.0f}% hot zones"),
    (col2, "#b8ff57", "HOT ZONES", f"{(filtered['zone_class']=='🔥 Hot Zone').sum()}", f"of {len(filtered)} properties"),
    (col3, "#ffb347", "AVG RENTAL YIELD", f"{avg_yield:.1f}%", "annualized estimate"),
    (col4, "#00e5ff", "PRICE / SQFT", f"${avg_ppsf:.0f}", "filtered avg"),
    (col5, "#ff4f6b", "UNDERVALUED", f"{underval_count}", "potential picks"),
]
for col, accent, label, val, delta in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card" style="--card-accent:{accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺 Zone Intelligence",
    "📈 Trend Analysis",
    "🔮 Growth Predictor",
    "🏆 Hotspot Ranking",
    "📊 Market Matrix",
])

with tab1:
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">Growth Velocity Heatmap <span class="tag">Live</span></div>', unsafe_allow_html=True)

        neigh_agg = filtered.groupby('Neighborhood').agg(
            avg_gvs=('growth_velocity_score', 'mean'),
            avg_price=('price', 'mean'),
            avg_yield=('rental_yield', 'mean'),
            count=('price', 'count'),
            avg_ppsf=('price_per_sqft', 'mean'),
        ).reset_index().sort_values('avg_gvs', ascending=False)

        fig_heat = px.bar(
            neigh_agg,
            x='Neighborhood', y='avg_gvs',
            color='avg_gvs',
            color_continuous_scale=[(0,'#0d1f3c'),(0.4,'#0096c7'),(0.7,'#00e5ff'),(1,'#b8ff57')],
            hover_data={'avg_price': ':,.0f', 'avg_yield': ':.2f', 'count': True, 'avg_ppsf': ':,.0f'},
            labels={'avg_gvs': 'Growth Velocity Score', 'Neighborhood': ''}
        )
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono, monospace', color='#6b7fa3', size=9),
            xaxis=dict(tickangle=-45, gridcolor='#1a2d4a', showgrid=False),
            yaxis=dict(gridcolor='#1a2d4a', range=[0, 10]),
            coloraxis_showscale=False,
            margin=dict(t=10, b=60, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-header">Price vs Growth Velocity <span class="tag">Scatter</span></div>', unsafe_allow_html=True)

        fig_scatter = px.scatter(
            filtered.sample(min(len(filtered), 500), random_state=42),
            x='price', y='growth_velocity_score',
            color='zone_class',
            size='GrLivArea', size_max=18,
            hover_data=['Neighborhood', 'price_per_sqft', 'rental_yield'],
            color_discrete_map={
                "🔥 Hot Zone": "#b8ff57",
                "🌤 Warm Zone": "#ffb347",
                "🌊 Stable Zone": "#00e5ff",
                "❄️ Cool Zone": "#4a6080",
            },
            labels={'price': 'Sale Price ($)', 'growth_velocity_score': 'Growth Score (0-10)'}
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono, monospace', color='#6b7fa3', size=9),
            xaxis=dict(gridcolor='#1a2d4a'),
            yaxis=dict(gridcolor='#1a2d4a'),
            legend=dict(orientation='h', y=-0.18, font=dict(size=9)),
            margin=dict(t=10, b=60, l=10, r=10),
            height=310,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with right:
        st.markdown('<div class="section-header">Top Neighbourhoods <span class="tag">Ranked</span></div>', unsafe_allow_html=True)

        for _, row in neigh_agg.head(8).iterrows():
            score = row['avg_gvs']
            cls = "score-hot" if score >= 7 else ("score-warm" if score >= 5 else "score-cool")
            st.markdown(f"""
            <div class="zone-card">
                <div>
                    <div class="zone-name">{row['Neighborhood']}</div>
                    <div class="zone-meta">
                        ${row['avg_price']:,.0f} avg &nbsp;|&nbsp;
                        {row['avg_yield']*100:.1f}% yield &nbsp;|&nbsp;
                        {row['count']} listings
                    </div>
                </div>
                <div class="zone-score {cls}">{score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1rem;">Zone Distribution</div>', unsafe_allow_html=True)
        zone_counts = filtered['zone_class'].value_counts().reset_index()
        zone_counts.columns = ['Zone', 'Count']
        fig_pie = px.pie(
            zone_counts, names='Zone', values='Count',
            color='Zone',
            color_discrete_map={
                "🔥 Hot Zone": "#b8ff57", "🌤 Warm Zone": "#ffb347",
                "🌊 Stable Zone": "#00e5ff", "❄️ Cool Zone": "#4a6080"
            },
            hole=0.6,
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Space Mono', color='#6b7fa3', size=9),
            legend=dict(orientation='h', y=-0.1, font=dict(size=8)),
            margin=dict(t=0, b=40, l=0, r=0), height=230,
            showlegend=True,
        )
        fig_pie.update_traces(textinfo='percent', textfont_size=10)
        st.plotly_chart(fig_pie, use_container_width=True)


with tab2:
    st.markdown('<div class="section-header">Rental Yield vs Selling Price — Undervaluation Matrix <span class="tag">Alpha Signal</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    ◈ INSIGHT &nbsp;·&nbsp; Properties in the <strong>top-left quadrant</strong> (high rental yield, lower price) are
    prime candidates for appreciation. These represent zones where demand outpaces
    current valuations — the classic lead indicator for a growth hotspot.
    </div>
    """, unsafe_allow_html=True)

    fig_matrix = px.scatter(
        filtered.sample(min(len(filtered), 600), random_state=7),
        x='price', y='rental_yield',
        color='growth_velocity_score',
        size='GrLivArea', size_max=14,
        hover_name='Neighborhood',
        hover_data={'price': ':,.0f', 'rental_yield': ':.3f', 'growth_velocity_score': ':.2f'},
        color_continuous_scale=[(0,'#0d1f3c'), (0.3,'#0096c7'), (0.7,'#00e5ff'), (1,'#b8ff57')],
        labels={'price': 'Sale Price ($)', 'rental_yield': 'Rental Yield (annual)', 'growth_velocity_score': 'GV Score'}
    )
    fig_matrix.add_hline(y=filtered['rental_yield'].median(), line_dash='dash',
                          line_color='rgba(255,179,71,0.4)', annotation_text='Median Yield',
                          annotation_font_color='#ffb347', annotation_font_size=9)
    fig_matrix.add_vline(x=filtered['price'].median(), line_dash='dash',
                          line_color='rgba(0,229,255,0.3)', annotation_text='Median Price',
                          annotation_font_color='#00e5ff', annotation_font_size=9)
    fig_matrix.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
        font=dict(family='Space Mono, monospace', color='#6b7fa3', size=9),
        xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='#1a2d4a'),
        height=380, margin=dict(t=10, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Infrastructure Score vs Growth <span class="tag">Lead Indicator</span></div>', unsafe_allow_html=True)
        fig_infra = px.scatter(
            filtered, x='infra_score', y='growth_velocity_score',
            color='zone_class',
            color_discrete_map={
                "🔥 Hot Zone": "#b8ff57", "🌤 Warm Zone": "#ffb347",
                "🌊 Stable Zone": "#00e5ff", "❄️ Cool Zone": "#4a6080"
            },
            labels={'infra_score': 'Infrastructure Score', 'growth_velocity_score': 'Growth Velocity Score'}
        )
        fig_infra.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=9),
            xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='#1a2d4a'),
            legend=dict(font=dict(size=8)), height=300, margin=dict(t=10, b=10)
        )
        st.plotly_chart(fig_infra, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Price Appreciation Potential <span class="tag">60-Month</span></div>', unsafe_allow_html=True)
        proj_df = filtered.groupby('Neighborhood').agg(
            current=('price', 'mean'),
            proj_24=('proj_24m', 'mean'),
            proj_60=('proj_60m', 'mean')
        ).reset_index().sort_values('proj_60', ascending=False).head(10)

        fig_proj = go.Figure()
        fig_proj.add_trace(go.Bar(name='Current Price', x=proj_df['Neighborhood'], y=proj_df['current'],
                                   marker_color='rgba(0,150,199,0.5)'))
        fig_proj.add_trace(go.Bar(name='24-Month Proj.', x=proj_df['Neighborhood'], y=proj_df['proj_24'],
                                   marker_color='rgba(0,229,255,0.7)'))
        fig_proj.add_trace(go.Bar(name='60-Month Proj.', x=proj_df['Neighborhood'], y=proj_df['proj_60'],
                                   marker_color='rgba(184,255,87,0.85)'))
        fig_proj.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=8),
            xaxis=dict(tickangle=-35, gridcolor='#1a2d4a'),
            yaxis=dict(gridcolor='#1a2d4a'),
            legend=dict(orientation='h', y=-0.25, font=dict(size=8)),
            height=300, margin=dict(t=10, b=70, l=10, r=10)
        )
        st.plotly_chart(fig_proj, use_container_width=True)

    st.markdown('<div class="section-header">Seasonal Market Activity (by Month Sold)</div>', unsafe_allow_html=True)
    monthly = filtered.groupby('MoSold').agg(
        avg_price=('price', 'mean'), count=('price', 'count'),
        avg_gvs=('growth_velocity_score', 'mean')
    ).reset_index()
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly['Month'] = monthly['MoSold'].map(month_names)

    fig_season = make_subplots(specs=[[{"secondary_y": True}]])
    fig_season.add_trace(go.Bar(x=monthly['Month'], y=monthly['count'], name='Transactions',
                                 marker_color='rgba(0,150,199,0.5)'), secondary_y=False)
    fig_season.add_trace(go.Scatter(x=monthly['Month'], y=monthly['avg_price'], name='Avg Price',
                                     line=dict(color='#b8ff57', width=2.5), mode='lines+markers',
                                     marker=dict(size=7)), secondary_y=True)
    fig_season.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
        font=dict(family='Space Mono', color='#6b7fa3', size=9),
        legend=dict(orientation='h', y=-0.2, font=dict(size=9)),
        height=260, margin=dict(t=10, b=60, l=10, r=10),
    )
    fig_season.update_xaxes(gridcolor='#1a2d4a')
    fig_season.update_yaxes(gridcolor='#1a2d4a', title_text='Transactions', secondary_y=False)
    fig_season.update_yaxes(title_text='Avg Price ($)', secondary_y=True)
    st.plotly_chart(fig_season, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">AI Growth Velocity Predictor <span class="tag">GBT Model</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    ◈ Enter property parameters below to compute a predicted Growth Velocity Score (0–10).
    Scores above 7 indicate a <strong>High-Growth Zone</strong> with strong investment potential
    over a 24–60 month horizon.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        inp_price = st.number_input("Property Price ($)", min_value=50000, max_value=800000, value=220000, step=5000)
        inp_infra = st.slider("Infrastructure Score", 1.0, 10.0, 6.0, 0.1,
                               help="Garage, bathrooms, basement quality proxy")
        inp_rent_yield = st.slider("Rental Yield (annual %)", 0.5, 12.0, 7.2, 0.1,
                                    help="Annual rent / sale price × 100")
    with c2:
        inp_listing = st.slider("Listing Density (market activity)", 0.0, 10.0, 5.0, 0.5)
        inp_velocity = st.slider("Pricing Velocity (vs neighbourhood)", -0.3, 0.8, 0.1, 0.05,
                                  help="How much above/below neighbourhood median")
        inp_reno = st.slider("Renovation Recency Score", 0.0, 1.0, 0.5, 0.05)
    with c3:
        inp_age = st.number_input("Building Age (years)", min_value=1, max_value=130, value=25)
        inp_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=5000, value=1500, step=50)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Compute Growth Score", use_container_width=True)

    if predict_btn:
        input_vec = np.array([[
            inp_price, inp_infra, inp_rent_yield / 100,
            inp_listing, inp_velocity, inp_reno, inp_age, inp_area
        ]])
        pred_score = float(model.predict(input_vec)[0])
        pred_score = max(0, min(10, pred_score))

        zone_label = ("🔥 HOT ZONE — Strong Buy Signal" if pred_score >= 7
                      else "🌤 WARM ZONE — Moderate Potential" if pred_score >= 5
                      else "🌊 STABLE — Hold / Watch" if pred_score >= 3
                      else "❄️ COOL ZONE — Low Priority")
        score_color = ("#b8ff57" if pred_score >= 7 else
                       "#ffb347" if pred_score >= 5 else
                       "#00e5ff" if pred_score >= 3 else "#4a6080")

        proj24 = inp_price * (1 + pred_score * 0.012)
        proj60 = inp_price * (1 + pred_score * 0.030)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(13,20,35,0.8));
                    border: 2px solid {score_color}30; border-radius: 14px; padding: 2rem;
                    margin: 1.5rem 0; text-align: center; position: relative; overflow: hidden;">
            <div style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.2em;
                        text-transform:uppercase; color:#6b7fa3; margin-bottom:0.8rem;">
                PREDICTED GROWTH VELOCITY SCORE
            </div>
            <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:5rem;
                        color:{score_color}; line-height:1; letter-spacing:-0.02em;">
                {pred_score:.2f}
            </div>
            <div style="font-size:0.7rem; color:#6b7fa3; font-family:'Space Mono',monospace;">/ 10.0 &nbsp; | &nbsp; R² = {r2:.3f}</div>
            <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.1rem;
                        color:{score_color}; margin-top:1rem; letter-spacing:0.05em;">
                {zone_label}
            </div>
            <div style="display:flex; justify-content:center; gap:3rem; margin-top:1.5rem;">
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#6b7fa3; letter-spacing:0.1em; text-transform:uppercase;">24-Month Projection</div>
                    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.3rem; color:#00e5ff;">${proj24:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#6b7fa3; letter-spacing:0.1em; text-transform:uppercase;">60-Month Projection</div>
                    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.3rem; color:#b8ff57;">${proj60:,.0f}</div>
                </div>
                <div>
                    <div style="font-family:'Space Mono',monospace; font-size:0.6rem; color:#6b7fa3; letter-spacing:0.1em; text-transform:uppercase;">Estimated Return</div>
                    <div style="font-family:'Syne',sans-serif; font-weight:700; font-size:1.3rem; color:#ffb347;">{((proj60 - inp_price) / inp_price) * 100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fi = pd.DataFrame({
            'Feature': ['Price', 'Infra Score', 'Rental Yield', 'Listing Density',
                        'Price Velocity', 'Reno Score', 'Age', 'Living Area'],
            'Importance': model.feature_importances_
        }).sort_values('Importance')

        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                        color='Importance',
                        color_continuous_scale=[(0,'#0d1f3c'),(0.5,'#0096c7'),(1,'#b8ff57')])
        fig_fi.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono, monospace', color='#6b7fa3', size=9),
            xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='rgba(0,0,0,0)'),
            coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10), height=250,
            title=dict(text='Feature Importance', font=dict(color='#6b7fa3', size=10))
        )
        st.plotly_chart(fig_fi, use_container_width=True)


with tab4:
    st.markdown('<div class="section-header">Investment Hotspot Leaderboard <span class="tag">Top Picks</span></div>', unsafe_allow_html=True)

    n_top = st.slider("Show top N hotspots", 5, 50, 20)
    sort_by = st.selectbox("Rank by", ["growth_velocity_score", "rental_yield",
                                         "underval_score", "pricing_velocity", "price_per_sqft"])

    top_props = (
        filtered[filtered['zone_class'].isin(["🔥 Hot Zone", "🌤 Warm Zone"])]
        .sort_values(sort_by, ascending=False)
        .head(n_top)
        [['Neighborhood', 'price', 'GrLivArea', 'price_per_sqft',
          'rental_yield', 'infra_score', 'growth_velocity_score',
          'zone_class', 'proj_24m', 'proj_60m', 'underval_score']]
        .copy()
    )

    top_props.columns = [
        'Neighbourhood', 'Price ($)', 'Area (sqft)', 'Price/sqft',
        'Rental Yield', 'Infra Score', 'GV Score', 'Zone',
        'Proj 24M ($)', 'Proj 60M ($)', 'Underval Score'
    ]
    top_props['Price ($)'] = top_props['Price ($)'].map('${:,.0f}'.format)
    top_props['Proj 24M ($)'] = top_props['Proj 24M ($)'].map('${:,.0f}'.format)
    top_props['Proj 60M ($)'] = top_props['Proj 60M ($)'].map('${:,.0f}'.format)
    top_props['Price/sqft'] = top_props['Price/sqft'].map('${:.0f}'.format)
    top_props['Rental Yield'] = top_props['Rental Yield'].map('{:.2%}'.format)
    top_props['GV Score'] = top_props['GV Score'].map('{:.2f}'.format)
    top_props['Infra Score'] = top_props['Infra Score'].map('{:.1f}'.format)
    top_props['Underval Score'] = top_props['Underval Score'].map('{:.3f}'.format)

    st.dataframe(
        top_props,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Neighbourhood Comparison Radar</div>', unsafe_allow_html=True)

    top_neigh = (filtered.groupby('Neighbourhood' if 'Neighbourhood' in filtered.columns else 'Neighborhood')
                 .agg(gvs=('growth_velocity_score','mean'), infra=('infra_score','mean'),
                      yield_=('rental_yield','mean'), density=('listing_density','mean'),
                      underval=('underval_score','mean'))
                 .reset_index()
                 .nlargest(6, 'gvs'))

    top_neigh.columns = ['Neighbourhood','GV Score','Infra','Yield','Density','Underval']
    scaler_r = MinMaxScaler()
    radar_vals = scaler_r.fit_transform(top_neigh[['GV Score','Infra','Yield','Density','Underval']])
    cats = ['GV Score', 'Infrastructure', 'Rental Yield', 'Listing Density', 'Undervaluation']
    colors_r = ['#b8ff57','#00e5ff','#ffb347','#ff4f6b','#a78bfa','#34d399']

    fig_radar = go.Figure()
    for i, row in enumerate(radar_vals):
        vals = list(row) + [row[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill='toself', name=top_neigh.iloc[i]['Neighbourhood'],
            line_color=colors_r[i % len(colors_r)],
            fillcolor=colors_r[i % len(colors_r)],
            opacity=0.8,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(13,20,35,0.6)',
            radialaxis=dict(visible=True, range=[0,1], gridcolor='#1a2d4a',
                             tickfont=dict(color='#6b7fa3', size=8)),
            angularaxis=dict(gridcolor='#1a2d4a', tickfont=dict(color='#aab8cc', size=9))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono', color='#6b7fa3'),
        legend=dict(orientation='h', y=-0.15, font=dict(size=9)),
        height=380, margin=dict(t=30, b=80, l=50, r=50)
    )
    st.plotly_chart(fig_radar, use_container_width=True)


with tab5:
    st.markdown('<div class="section-header">Multi-Dimensional Market Matrix <span class="tag">Analytics</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Box plot: price distribution by zone
        fig_box = px.box(
            filtered, x='zone_class', y='price',
            color='zone_class',
            color_discrete_map={
                "🔥 Hot Zone": "#b8ff57", "🌤 Warm Zone": "#ffb347",
                "🌊 Stable Zone": "#00e5ff", "❄️ Cool Zone": "#4a6080"
            },
            labels={'zone_class': 'Zone', 'price': 'Sale Price ($)'},
            title='Price Distribution by Zone'
        )
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=9),
            showlegend=False, height=300,
            margin=dict(t=40, b=20, l=10, r=10),
            title_font=dict(color='#aab8cc', size=11),
            xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='#1a2d4a')
        )
        st.plotly_chart(fig_box, use_container_width=True)

        fig_violin = px.violin(
            filtered, x='BldgType', y='growth_velocity_score',
            color='BldgType', box=True,
            labels={'BldgType': 'Building Type', 'growth_velocity_score': 'GV Score'},
            title='Growth Score by Building Type'
        )
        fig_violin.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=9),
            showlegend=False, height=300,
            margin=dict(t=40, b=20, l=10, r=10),
            title_font=dict(color='#aab8cc', size=11),
            xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='#1a2d4a')
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    with col2:
        
        corr_cols = ['price', 'growth_velocity_score', 'infra_score',
                      'rental_yield', 'listing_density', 'pricing_velocity',
                      'age', 'GrLivArea', 'price_per_sqft']
        corr_matrix = filtered[corr_cols].corr().round(2)
        corr_labels = ['Price', 'GV Score', 'Infra', 'Yield', 'Density',
                        'Velocity', 'Age', 'Area', 'PPSF']

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_labels, y=corr_labels,
            colorscale=[(0,'#ff4f6b'), (0.5,'#0d1f3c'), (1,'#b8ff57')],
            zmid=0, text=corr_matrix.values.round(2),
            texttemplate='%{text}', textfont_size=8,
            hoverongaps=False,
        ))
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=8),
            title=dict(text='Correlation Matrix', font=dict(color='#aab8cc', size=11)),
            height=310, margin=dict(t=40, b=20, l=10, r=10),
            xaxis=dict(tickangle=-35),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        
        fig_hist = px.histogram(
            filtered, x='growth_velocity_score', nbins=30,
            color_discrete_sequence=['#00e5ff'],
            labels={'growth_velocity_score': 'Growth Velocity Score'},
            title='Growth Score Distribution'
        )
        fig_hist.add_vline(x=filtered['growth_velocity_score'].mean(),
                            line_dash='dash', line_color='#b8ff57',
                            annotation_text=f"Mean: {filtered['growth_velocity_score'].mean():.2f}",
                            annotation_font_color='#b8ff57', annotation_font_size=9)
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,20,35,0.6)',
            font=dict(family='Space Mono', color='#6b7fa3', size=9),
            height=280, margin=dict(t=40, b=20, l=10, r=10),
            title_font=dict(color='#aab8cc', size=11),
            xaxis=dict(gridcolor='#1a2d4a'), yaxis=dict(gridcolor='#1a2d4a'),
            bargap=0.05,
        )
        fig_hist.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig_hist, use_container_width=True)

    top_neigh_name = df.groupby('Neighborhood')['growth_velocity_score'].mean().idxmax()
    st.markdown(f"""
    <div class="insight-box" style="margin-top:1rem;">
    ◈ MARKET SUMMARY &nbsp;·&nbsp;
    Analysed <strong>{len(filtered):,} properties</strong> across <strong>{filtered['Neighborhood'].nunique()} neighbourhoods</strong>.
    Average Growth Velocity Score: <strong>{avg_gvs:.2f}/10</strong>.
    Top performing neighbourhood in full dataset: <strong>{top_neigh_name}</strong>.
    <br><br>
    ◈ RECOMMENDATION &nbsp;·&nbsp;
    Focus acquisition on <strong>Hot Zones</strong> with rental yields above the median ({filtered['rental_yield'].median()*100:.1f}%)
    and infrastructure scores ≥ 7.  These properties show the highest correlation with
    60-month price appreciation.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:3rem; border-top:1px solid #1a2d4a; padding:1.5rem 0 0.5rem;
            font-family:'Space Mono',monospace; font-size:0.6rem; color:#3a4d6a;
            text-align:center; letter-spacing:0.1em;">
URBANPULSE GROWTH INTELLIGENCE ENGINE &nbsp;·&nbsp;
Predictive Geospatial Analytics &nbsp;·&nbsp;
Model: Gradient Boosting Regressor &nbsp;·&nbsp;
Data: Ames Housing Dataset
</div>
""", unsafe_allow_html=True)