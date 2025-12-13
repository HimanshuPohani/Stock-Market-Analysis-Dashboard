import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import time

st.set_page_config(
    page_title="Stock Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
    <style>
    /* Main theme */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Neon glowing text */
    .neon-text {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        animation: pulsate 1.5s infinite alternate;
        margin-bottom: 2rem;
    }
    
    @keyframes pulsate {
        0% { text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff; }
        100% { text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 15px #00ffff, 0 0 20px #00ffff; }
    }
    
    /* Company name display */
    .company-display {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #ff00ff, #00ffff, #ffff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: rainbow 3s ease infinite;
        margin: 1rem 0;
        padding: 1rem;
        border: 2px solid transparent;
        border-image: linear-gradient(45deg, #ff00ff, #00ffff, #ffff00) 1;
    }
    
    @keyframes rainbow {
        0% { filter: hue-rotate(0deg); }
        100% { filter: hue-rotate(360deg); }
    }
    
    /* Animated buttons */
    .stButton>button {
        background: linear-gradient(45deg, #ff00ff, #00ffff);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        animation: buttonGlow 2s infinite;
    }
    
    @keyframes buttonGlow {
        0%, 100% { box-shadow: 0 0 10px #ff00ff; }
        50% { box-shadow: 0 0 20px #00ffff, 0 0 30px #00ffff; }
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(45deg, #00ffff, #ff00ff);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #00ffff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Custom animated header */
    .animated-header {
        background: linear-gradient(90deg, #ff00ff, #00ffff, #ffff00, #ff00ff);
        background-size: 400% 400%;
        animation: gradientShift 3s ease infinite;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Upload section styling */
    .upload-section {
        text-align: center;
        padding: 3rem;
        border: 2px dashed #00ffff;
        border-radius: 15px;
        margin: 2rem 0;
        background: rgba(0, 255, 255, 0.1);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2024-01-31', freq='B')
    tickers = ['TECH_A', 'TECH_B', 'FINANCE_A', 'HEALTH_A', 'ENERGY_A', 'CONSUMER_A']
    sectors = ['Technology', 'Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    data = []
    for ticker, sector in zip(tickers, sectors):
        prices = [100]
        for i in range(1, len(dates)):
            returns = np.random.normal(0.0005, 0.02)
            new_price = prices[-1] * (1 + returns)
            prices.append(new_price)
        
        for i, date in enumerate(dates):
            data.append({
                'Date': date,
                'Ticker': ticker,
                'Sector': sector,
                'Open': prices[i] * 0.99,
                'High': prices[i] * 1.02,
                'Low': prices[i] * 0.98,
                'Close': prices[i],
                'Volume': np.random.randint(1000000, 10000000),
                'Adj Close': prices[i]
            })
    
    return pd.DataFrame(data)

def calculate_technical_indicators(df):
    df = df.copy()
  
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Add moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    return df

def create_animated_header():
    st.markdown("""
    <div class="animated-header">
        <h1 style="text-align: center; color: white; margin: 0;">🚀 STOCK ANALYTICS DASHBOARD 🌟</h1>
        <p style="text-align: center; color: white; margin: 0;">Advanced Financial Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

def show_upload_section():
    st.markdown("""
    <div class="upload-section">
        <div class="upload-icon">📁</div>
        <h2>Upload Your Stock Dataset</h2>
        <p>Please upload your CSV file to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            st.success(f"✅ Dataset loaded successfully! Found {len(df)} records.")
            
            with st.expander("📊 Dataset Preview"):
                st.dataframe(df.head(10))
                
            return df
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    else:
        st.info("👆 Please upload a CSV file with stock data to proceed")
        return None

def show_technical_tab(company_data):
    st.markdown("### 📊 Technical Analysis")
    st.info("**What this shows:** Trading signals and price patterns for short-term decisions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_rsi = company_data['RSI'].iloc[-1] if not pd.isna(company_data['RSI'].iloc[-1]) else 50
        rsi_signal = "🔴 SELL" if current_rsi > 70 else "🟢 BUY" if current_rsi < 30 else "⚪ HOLD"
        st.metric("RSI Signal", f"{current_rsi:.1f}", rsi_signal)
        st.caption("RSI < 30: Oversold (Buy) | RSI > 70: Overbought (Sell)")
    
    with col2:
        macd = company_data['MACD'].iloc[-1] if not pd.isna(company_data['MACD'].iloc[-1]) else 0
        macd_signal = company_data['MACD_Signal'].iloc[-1] if not pd.isna(company_data['MACD_Signal'].iloc[-1]) else 0
        macd_trend = "🟢 BULLISH" if macd > macd_signal else "🔴 BEARISH"
        st.metric("MACD Trend", macd_trend)
        st.caption("MACD above Signal: Bullish | Below: Bearish")
    
    with col3:
        ma_20 = company_data['MA20'].iloc[-1] if not pd.isna(company_data['MA20'].iloc[-1]) else company_data['Close'].iloc[-1]
        ma_50 = company_data['MA50'].iloc[-1] if not pd.isna(company_data['MA50'].iloc[-1]) else company_data['Close'].iloc[-1]
        ma_signal = "🟢 UPTREND" if ma_20 > ma_50 else "🔴 DOWNTREND"
        st.metric("Trend Direction", ma_signal)
        st.caption("20MA > 50MA: Uptrend | 20MA < 50MA: Downtrend")
    
    with col4:
        current_price = company_data['Close'].iloc[-1]
        bb_upper = company_data['BB_Upper'].iloc[-1] if not pd.isna(company_data['BB_Upper'].iloc[-1]) else current_price
        bb_lower = company_data['BB_Lower'].iloc[-1] if not pd.isna(company_data['BB_Lower'].iloc[-1]) else current_price
        bb_position = "🔴 HIGH" if current_price > bb_upper else "🟢 LOW" if current_price < bb_lower else "⚪ NORMAL"
        st.metric("Price Level", bb_position)
        st.caption("Near upper band: High price | Near lower band: Low price")

def show_fundamental_tab(company_data):
    st.markdown("### 📈 Fundamental Analysis")
    st.info("**What this shows:** Company value and long-term investment potential")
    
    current_price = company_data['Close'].iloc[-1]
    avg_volume = company_data['Volume'].mean()
    volatility = company_data['Close'].pct_change().std() * np.sqrt(252) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💰 Value Metrics")
        # Simulated fundamental data
        pe_ratio = max(8, 25 - volatility * 0.5)
        st.metric("P/E Ratio", f"{pe_ratio:.1f}x")
        st.caption("Lower = Better value")
        
        market_cap = current_price * (avg_volume / 100) * 1000
        st.metric("Market Cap", f"${market_cap:,.0f}")
        st.caption("Company size")
        
        pb_ratio = pe_ratio * 0.7
        st.metric("P/B Ratio", f"{pb_ratio:.1f}x")
        st.caption("Price vs Book value")
    
    with col2:
        st.subheader("📊 Performance")
        total_return = ((company_data['Close'].iloc[-1] - company_data['Close'].iloc[0]) / company_data['Close'].iloc[0]) * 100
        
        days = (company_data['Date'].iloc[-1] - company_data['Date'].iloc[0]).days
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100 if days > 0 else total_return
        
        st.metric("Total Return", f"{total_return:.2f}%")
        st.caption("Overall performance")
        
        st.metric("Annual Return", f"{annual_return:.2f}%")
        st.caption("Yearly average")
        
        st.metric("Volatility", f"{volatility:.2f}%")
        st.caption("Risk level")
    
    with col3:
        st.subheader("⚡ Health Check")
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
        st.caption("Trading activity")
        
        volume_ratio = (avg_volume / 1000000)
        st.metric("Liquidity", f"{volume_ratio:.1f}M")
        st.caption("Ease of trading")
        
        stability = "🛡️ STABLE" if volatility < 15 else "🌊 VOLATILE" if volatility < 30 else "⚡ RISKY"
        st.metric("Stability", stability)
        st.caption("Price consistency")

def show_advanced_tab(company_data):
    st.markdown("### 🎯 Advanced Analytics")
    st.info("**What this shows:** Deep patterns and statistical insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Statistics")
        returns = company_data['Close'].pct_change().dropna()
        
        skewness = returns.skew()
        st.metric("Skewness", f"{skewness:.3f}")
        st.caption("Positive: More gains | Negative: More losses")
        
        kurtosis = returns.kurtosis()
        st.metric("Kurtosis", f"{kurtosis:.3f}")
        st.caption("High: More extreme moves")
        
        is_normal = "NORMAL" if abs(skewness) < 1 and abs(kurtosis) < 1 else "ABNORMAL"
        st.metric("Distribution", is_normal)
        st.caption("Pattern type")
    
    with col2:
        st.subheader("🔍 Patterns")
        short_trend = "🟢 UP" if company_data['Close'].iloc[-1] > company_data['Close'].iloc[-5] else "🔴 DOWN"
        st.metric("5-Day Trend", short_trend)
        st.caption("Recent direction")
        
        long_trend = "🟢 UP" if company_data['Close'].iloc[-1] > company_data['Close'].iloc[-20] else "🔴 DOWN"
        st.metric("20-Day Trend", long_trend)
        st.caption("Long-term direction")
        
        recent_vol = company_data['Close'].pct_change().tail(20).std() * 100
        vol_status = "LOW" if recent_vol < 1.5 else "MEDIUM" if recent_vol < 3 else "HIGH"
        st.metric("Recent Volatility", vol_status)
        st.caption("Current risk level")

def show_comparative_tab(df, selected_company, start_date, end_date):
    st.markdown("### 🔍 Comparative Analysis")
    st.info("**What this shows:** How this stock compares to others")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        compare_companies = st.multiselect(
            "Compare with:",
            options=[c for c in df['Ticker'].unique() if c != selected_company],
            default=[],
            max_selections=3
        )
    
    if compare_companies:
        compare_data = df[(df['Ticker'].isin([selected_company] + compare_companies)) & 
                         (df['Date'] >= pd.to_datetime(start_date)) & 
                         (df['Date'] <= pd.to_datetime(end_date))]
        
        # Normalized prices for comparison
        compare_pivot = compare_data.pivot(index='Date', columns='Ticker', values='Close')
        compare_normalized = compare_pivot / compare_pivot.iloc[0] * 100
        
        fig_compare = px.line(compare_normalized.reset_index(), x='Date', y=compare_normalized.columns,
                             title="Price Comparison (All start at 100)", template="plotly_dark")
        fig_compare.update_layout(height=400)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Performance table - FIXED SECTION
        st.subheader("Performance Summary")
        performance_data = []
        
        # First collect all data
        for ticker in [selected_company] + compare_companies:
            ticker_data = compare_data[compare_data['Ticker'] == ticker]
            if len(ticker_data) > 1:
                total_return = ((ticker_data['Close'].iloc[-1] - ticker_data['Close'].iloc[0]) / ticker_data['Close'].iloc[0]) * 100
                volatility = ticker_data['Close'].pct_change().std() * np.sqrt(252) * 100
                performance_data.append({
                    'Stock': ticker,
                    'Return_Value': total_return,  # Store numeric value for comparison
                    'Risk': volatility,
                })
        
        # Now calculate status after all data is collected
        if performance_data:
            max_return = max([d['Return_Value'] for d in performance_data])
            
            # Format the final display data
            display_data = []
            for d in performance_data:
                status = '🟢 BEST' if d['Return_Value'] == max_return else '⚪ GOOD' if d['Return_Value'] > 0 else '🔴 POOR'
                display_data.append({
                    'Stock': d['Stock'],
                    'Return %': f"{d['Return_Value']:.2f}%",
                    'Risk': f"{d['Risk']:.2f}%",
                    'Status': status
                })
            
            perf_df = pd.DataFrame(display_data)
            st.dataframe(perf_df, use_container_width=True)
    else:
        st.info("👆 Select other stocks to compare with " + selected_company)

def show_dashboard(df):
    with st.sidebar:
        st.markdown("## 🔮 Control Panel")
        
        selected_company = st.selectbox(
            "SELECT COMPANY",
            options=sorted(df['Ticker'].unique()),
            index=0,
            help="Choose a company to analyze"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "START DATE",
                value=df['Date'].min().date(),
                max_value=df['Date'].max().date()
            )
        with col2:
            end_date = st.date_input(
                "END DATE",
                value=df['Date'].max().date(),
                min_value=df['Date'].min().date()
            )
        
        st.markdown("---")
        st.markdown("### 🎨 Theme Settings")
        show_animations = st.toggle("Enable Animations", True)
        auto_refresh = st.toggle("Live Updates", False)
        
        if auto_refresh:
            st.warning("Live mode enabled - data updates every 30 seconds")
            time.sleep(30)
            st.rerun()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f'<div class="company-display">🎯 {selected_company} 📈</div>', unsafe_allow_html=True)
    
    # Filter data based on selection
    company_data = df[(df['Ticker'] == selected_company) & 
                     (df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))].copy()
    
    company_data = calculate_technical_indicators(company_data)
    
    # KEY METRICS (Always visible)
    st.markdown("## 📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = company_data['Close'].iloc[-1]
        price_change = ((current_price - company_data['Close'].iloc[0]) / company_data['Close'].iloc[0]) * 100
        st.metric(
            "Current Price", 
            f"${current_price:.2f}",
            f"{price_change:+.2f}%"
        )
    
    with col2:
        volatility = company_data['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.2f}%")
    
    with col3:
        total_return = ((company_data['Close'].iloc[-1] - company_data['Close'].iloc[0]) / company_data['Close'].iloc[0]) * 100
        st.metric("Total Return", f"{total_return:.2f}%")
    
    with col4:
        avg_volume = company_data['Volume'].mean()
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    with col5:
        rsi = company_data['RSI'].iloc[-1] if not pd.isna(company_data['RSI'].iloc[-1]) else 50
        st.metric("RSI", f"{rsi:.1f}")
    
    # ANALYSIS MODES IN TABS
    st.markdown("## 🎛️ Analysis Modes")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Chart", 
        "🔧 Technical", 
        "💰 Fundamental", 
        "🎯 Advanced", 
        "🔍 Compare"
    ])
    
    with tab1:
        # Main Price Chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, subplot_titles=('Price Chart', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        fig.add_trace(go.Candlestick(x=company_data['Date'],
                                    open=company_data['Open'],
                                    high=company_data['High'],
                                    low=company_data['Low'],
                                    close=company_data['Close'],
                                    name='Price'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=company_data['Date'], y=company_data['MA20'],
                                line=dict(color='#00ffff', width=2),
                                name='20-Day MA'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=company_data['Date'], y=company_data['MA50'],
                                line=dict(color='#ff00ff', width=2),
                                name='50-Day MA'), row=1, col=1)
        
        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' 
                 for _, row in company_data.iterrows()]
        
        fig.add_trace(go.Bar(x=company_data['Date'], y=company_data['Volume'],
                            name='Volume', marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark",
                         paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        show_technical_tab(company_data)
        
        # Technical charts
        col1, col2 = st.columns(2)
        with col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=company_data['Date'], y=company_data['RSI'],
                                        line=dict(color='#ffff00', width=2), name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI Indicator", height=300, template="plotly_dark")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=company_data['Date'], y=company_data['MACD'],
                                         line=dict(color='#00ffff', width=2), name='MACD'))
            fig_macd.add_trace(go.Scatter(x=company_data['Date'], y=company_data['MACD_Signal'],
                                         line=dict(color='#ff00ff', width=2), name='Signal'))
            fig_macd.update_layout(title="MACD Indicator", height=300, template="plotly_dark")
            st.plotly_chart(fig_macd, use_container_width=True)
    
    with tab3:
        show_fundamental_tab(company_data)
    
    with tab4:
        show_advanced_tab(company_data)
    
    with tab5:
        show_comparative_tab(df, selected_company, start_date, end_date)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Stock Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    load_css()
    create_animated_header()
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if st.session_state.df is None:
        df = show_upload_section()
        if df is not None:
            st.session_state.df = df
            st.rerun()
    else:
        show_dashboard(st.session_state.df)
        
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Upload New Dataset"):
            st.session_state.df = None
            st.rerun()

if __name__ == "__main__":
    main()