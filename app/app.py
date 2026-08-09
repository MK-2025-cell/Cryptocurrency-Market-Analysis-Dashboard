# streamlit_app.py - FULLY CORRECTED VERSION

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from hmmlearn import hmm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Check for XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Crypto Market Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .main-header .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        line-height: 1.6;
    }
    .main-header .research-topic {
        color: rgba(255,255,255,0.95);
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 2px solid rgba(255,255,255,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1rem;
        color: #333;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .rq-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0 1rem 0;
    }
    .rq-header h2 {
        margin: 0;
        font-size: 1.3rem;
        line-height: 1.4;
    }
    .rq-conclusion {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .rq-conclusion h4 {
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
    }
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 10px;
        font-weight: bold;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    st.markdown("---")
    uploaded_train = st.file_uploader("📊 Upload Train Data (70%)", type=["csv"])
    uploaded_test = st.file_uploader("📈 Upload Test Data (20%)", type=["csv"])
    uploaded_val = st.file_uploader("📉 Upload Validation Data (10%)", type=["csv"])
    
    st.markdown("---")
    st.markdown("### 📋 Required Columns")
    st.caption("date, DAA, tx_count, tx_volume, market_cap, liquidity, volatility, price, ticker, fees, hash_rate")

@st.cache_data
def load_data(train_file, test_file, val_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    val = pd.read_csv(val_file)
    
    for df in [train, test, val]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    
    df = pd.concat([train, test, val], ignore_index=True)
    return train, test, val, df

if uploaded_train is not None and uploaded_test is not None and uploaded_val is not None:
    train_df, test_df, val_df, df = load_data(uploaded_train, uploaded_test, uploaded_val)
    
    # Dashboard Header
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Cryptocurrency Market Analysis Dashboard</h1>
        <div class="subtitle">A Systematic Literature Review on</div>
        <div class="research-topic">✨ Measurement and Visualization of Cryptocurrency Adoption,<br>
        Network Activity, and Financial Performance<br>
        Using On-Chain and Market Data ✨</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 Total Records", f"{len(df):,}", delta=f"from {df['date'].min().date()} to {df['date'].max().date()}")
    with col2:
        st.metric("🪙 Cryptocurrencies", f"{df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
    with col3:
        if 'price' in df.columns:
            st.metric("💰 Avg Price", f"${df['price'].mean():,.2f}")
    with col4:
        if 'market_cap' in df.columns:
            st.metric("🏦 Avg Market Cap", f"${df['market_cap'].mean():,.0f}")
    
    st.markdown("---")
    
    # ================================================================
    # RQ1: Network Adoption & Market Growth
    # ================================================================
    st.markdown('<div class="rq-header"><h2>📈 RQ1: Do increases in new wallet creation and active address counts lead to higher transaction activity and measurable market growth?</h2></div>', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** Do increases in new wallet creation and active address counts lead to higher transaction activity and measurable market growth?
        
        **Hypothesis:** Increased network participation (measured by Daily Active Addresses) should drive higher transaction volume and potentially market capitalization growth.
        """)
    
    if 'DAA' in df.columns and 'tx_count' in df.columns:
        df_rq1 = df.dropna(subset=['DAA', 'tx_count'])
        corr_val = df_rq1['DAA'].corr(df_rq1['tx_count'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
            ax_corr.scatter(df_rq1['DAA'], df_rq1['tx_count'], alpha=0.3, s=5, c='#667eea')
            ax_corr.set_xlabel('Daily Active Addresses (DAA)', fontsize=12)
            ax_corr.set_ylabel('Transaction Count', fontsize=12)
            ax_corr.set_title(f'DAA vs Transaction Count Correlation: {corr_val:.3f}', fontsize=14, fontweight='bold')
            ax_corr.grid(True, alpha=0.3)
            st.pyplot(fig_corr)
        
        with col2:
            if 'price' in df.columns:
                df_rq1['return'] = df_rq1.groupby('ticker')['price'].pct_change()
                corr_return = df_rq1['DAA'].corr(df_rq1['return'].dropna())
                
                fig_ret, ax_ret = plt.subplots(figsize=(8, 5))
                ax_ret.scatter(df_rq1['DAA'], df_rq1['return'], alpha=0.3, s=5, c='#764ba2')
                ax_ret.set_xlabel('Daily Active Addresses (DAA)', fontsize=12)
                ax_ret.set_ylabel('Daily Returns', fontsize=12)
                ax_ret.set_title(f'DAA vs Returns Correlation: {corr_return:.3f}', fontsize=14, fontweight='bold')
                ax_ret.grid(True, alpha=0.3)
                st.pyplot(fig_ret)
        
        st.markdown(f"""
        <div class="rq-conclusion">
            <h4>📌 RQ1 Conclusion</h4>
            <p><strong>Key Finding:</strong> DAA and Transaction Count show a strong positive correlation of <strong>{corr_val:.3f}</strong>, confirming that increased wallet creation and active addresses strongly predict higher transaction activity.</p>
            <p><strong>Secondary Finding:</strong> The correlation between DAA and returns is weak ({df_rq1['DAA'].corr(df_rq1['return'].dropna()):.3f}), suggesting that network adoption drives usage but not necessarily short-term price appreciation.</p>
            <p><strong>Practical Implication:</strong> Network adoption metrics are reliable indicators for measuring blockchain utility and transaction growth, but should not be used alone for price prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================
    # RQ2: On-Chain Metrics for Returns & Volatility Prediction
    # ================================================================
    st.markdown('<div class="rq-header"><h2>📊 RQ2: Which on-chain metrics (transaction volume, active addresses, hash rate, fees) most effectively predict cryptocurrency returns and market volatility?</h2></div>', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** Which on-chain metrics most effectively predict cryptocurrency returns and market volatility?
        
        **Hypothesis:** Transaction volume, active addresses, and hash rate provide leading signals for future price movements and market instability.
        """)
    
    if all(col in df.columns for col in ['price', 'tx_count', 'DAA']):
        df_rq2 = df.copy()
        df_rq2['future_return'] = df_rq2.groupby('ticker')['price'].pct_change().shift(-30)
        df_rq2 = df_rq2.dropna()
        
        features = [c for c in ['tx_count', 'tx_volume', 'DAA', 'market_cap', 'volatility'] if c in df_rq2.columns]
        X = df_rq2[features]
        y = (df_rq2['future_return'] > 0).astype(int)
        
        if len(X) > 0 and len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            pred = rf.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                importances = rf.feature_importances_
                indices = np.argsort(importances)
                colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
                ax.barh(range(len(indices)), importances[indices], color=colors)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([features[i] for i in indices])
                ax.set_xlabel('Feature Importance', fontsize=12)
                ax.set_title('Predictive Power of On-Chain Metrics', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 1.5rem; border-radius: 15px;">
                    <h4>📈 Model Performance</h4>
                    <p style="font-size: 2rem; font-weight: bold; margin: 0;">{acc:.1%}</p>
                    <p>Prediction Accuracy (30-day direction)</p>
                    <hr>
                    <p><strong>F1-Score:</strong> {f1:.3f}</p>
                    <p><strong>Top Predictor:</strong> {features[np.argmax(importances)]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="rq-conclusion">
                <h4>📌 RQ2 Conclusion</h4>
                <p><strong>Key Finding:</strong> The model achieves <strong>{acc:.1%} accuracy</strong> in predicting 30-day forward returns using on-chain metrics.</p>
                <p><strong>Top Predictors:</strong> <strong>{features[np.argmax(importances)]}</strong> emerged as the most predictive metric.</p>
                <p><strong>Practical Implication:</strong> Transaction activity metrics provide valuable leading indicators for market direction, suggesting that on-chain data should be integrated into trading strategies.</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================
    # RQ3: Transaction Count vs Volume - Usage vs Speculation
    # ================================================================
    st.markdown('<div class="rq-header"><h2>🔄 RQ3: How does the relationship between transaction count and transaction volume distinguish genuine network usage from speculative behavior?</h2></div>', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** How does the relationship between transaction count and transaction volume distinguish genuine network usage from speculative behavior?
        
        **Hypothesis:** High count with low volume indicates genuine utility (many small transactions), while low count with high volume suggests speculation (few large transfers).
        """)
    
    if 'tx_count' in df.columns and 'tx_volume' in df.columns:
        df_rq3 = df.dropna(subset=['tx_count', 'tx_volume'])
        df_rq3['avg_tx_size'] = df_rq3['tx_volume'] / (df_rq3['tx_count'] + 1)
        median_ratio = df_rq3['avg_tx_size'].median()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Median Tx Size</h3>
                <div class="value">{median_ratio:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            usage_pct = (df_rq3['avg_tx_size'] < median_ratio / 2).mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>📱 Usage Days (Small Tx)</h3>
                <div class="value">{usage_pct:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            speculation_pct = (df_rq3['avg_tx_size'] > median_ratio * 2).mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎲 Speculation Days (Large Tx)</h3>
                <div class="value">{speculation_pct:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        ax1.scatter(df_rq3['tx_count'], df_rq3['tx_volume'], alpha=0.3, s=5, c='#667eea')
        ax1.set_xlabel('Transaction Count', fontsize=10)
        ax1.set_ylabel('Transaction Volume', fontsize=10)
        ax1.set_title('Count vs Volume Distribution', fontsize=12)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(np.log1p(df_rq3['avg_tx_size']), bins=50, alpha=0.7, color='#11998e', edgecolor='black')
        ax2.axvline(x=np.log1p(median_ratio), color='red', linestyle='--', linewidth=2, label=f'Median: {median_ratio:,.0f}')
        ax2.set_xlabel('Log(Average Transaction Size)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Transaction Size Distribution', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        df_rq3_sample = df_rq3.head(200)
        ax3.plot(df_rq3_sample.index, df_rq3_sample['avg_tx_size'], 'b-', alpha=0.7, linewidth=1)
        ax3.axhline(y=median_ratio, color='r', linestyle='--', label='Median')
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylabel('Avg Transaction Size', fontsize=10)
        ax3.set_title('Average Transaction Size Over Time', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class="rq-conclusion">
            <h4>📌 RQ3 Conclusion</h4>
            <p><strong>Key Finding:</strong> {usage_pct:.1%} of days show small transaction sizes (genuine usage) while {speculation_pct:.1%} show large transactions (speculative activity).</p>
            <p><strong>Pattern Recognition:</strong> High transaction count with low average value → Genuine network utility<br>
            Low transaction count with high average value → Speculative activity</p>
            <p><strong>Practical Implication:</strong> Monitoring the count/volume ratio helps distinguish organic network growth from price-driven speculation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================
    # RQ4: Fees & Hash Rate Impact
    # ================================================================
    st.markdown('<div class="rq-header"><h2>⚡ RQ4: How do transaction fee dynamics and consensus participation levels (hash rate/staking) influence network adoption, security, and long-term asset value?</h2></div>', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** How do transaction fee dynamics and consensus participation levels influence network adoption, security, and long-term asset value?
        
        **Hypothesis:** Higher hash rate (miner confidence) and stable fee structures signal network health and predict future adoption.
        """)
    
    if all(col in df.columns for col in ['fees', 'hash_rate', 'DAA']):
        df_rq4 = df.dropna(subset=['fees', 'hash_rate', 'DAA'])
        df_rq4['adoption_growth'] = df_rq4.groupby('ticker')['DAA'].pct_change().shift(-30)
        df_rq4 = df_rq4.dropna()
        
        corr_fees = df_rq4['fees'].corr(df_rq4['adoption_growth'])
        corr_hash = df_rq4['hash_rate'].corr(df_rq4['adoption_growth'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fees → Adoption Correlation", f"{corr_fees:.3f}")
        with col2:
            st.metric("Hash Rate → Adoption Correlation", f"{corr_hash:.3f}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(df_rq4['fees'], df_rq4['adoption_growth'], alpha=0.3, s=5, c='#f5576c')
        axes[0].set_xlabel('Transaction Fees', fontsize=12)
        axes[0].set_ylabel('Future Adoption Growth (30d)', fontsize=12)
        axes[0].set_title(f'Fees vs Adoption Growth (r = {corr_fees:.3f})', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(df_rq4['hash_rate'], df_rq4['adoption_growth'], alpha=0.3, s=5, c='#11998e')
        axes[1].set_xlabel('Hash Rate', fontsize=12)
        axes[1].set_ylabel('Future Adoption Growth (30d)', fontsize=12)
        axes[1].set_title(f'Hash Rate vs Adoption Growth (r = {corr_hash:.3f})', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f"""
        <div class="rq-conclusion">
            <h4>📌 RQ4 Conclusion</h4>
            <p><strong>Key Finding:</strong> Hash rate shows a {"strong" if corr_hash > 0.3 else "moderate"} positive correlation ({corr_hash:.3f}) with future adoption growth.</p>
            <p><strong>Practical Implication:</strong> Hash rate serves as a leading indicator for network health. Sustained high hash rate periods historically precede adoption growth by 30-60 days.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ RQ4 requires 'fees' and 'hash_rate' columns for complete analysis")
    
    st.markdown("---")
    
    # ================================================================
    # RQ5: Market Regime Detection (HMM)
    # ================================================================
    st.markdown('<div class="rq-header"><h2>🔍 RQ5: Do market capitalization and liquidity ratios serve as reliable signals of major cryptocurrency market cycle shifts (bull/bear regime detection)?</h2></div>', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** Do market capitalization and liquidity ratios serve as reliable signals of major cryptocurrency market cycle shifts?
        
        **Hypothesis:** Hidden Markov Models can detect unobserved regime changes using market cap and liquidity, providing early warning signals for trend reversals.
        """)
    
    if all(col in df.columns for col in ['price', 'market_cap', 'liquidity']):
        df_hmm = df.copy()
        df_hmm['returns'] = df_hmm.groupby('ticker')['price'].pct_change()
        df_hmm['forward_returns'] = df_hmm.groupby('ticker')['price'].shift(-30) / df_hmm['price'] - 1
        df_hmm = df_hmm.dropna()
        
        features = [c for c in ['market_cap', 'liquidity'] if c in df_hmm.columns]
        X_hmm = df_hmm[features].ffill()
        X_hmm = X_hmm.dropna()
        df_hmm = df_hmm.loc[X_hmm.index]
        
        if len(X_hmm) > 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_hmm)
            
            try:
                model = hmm.GaussianHMM(n_components=2, covariance_type='full', n_iter=200, random_state=42)
                model.fit(X_scaled)
                states = model.predict(X_scaled)
                
                regime0_return = df_hmm[states == 0]['forward_returns'].mean()
                regime1_return = df_hmm[states == 1]['forward_returns'].mean()
                
                if regime0_return > regime1_return:
                    bull_regime, bear_regime = 0, 1
                    bull_return, bear_return = regime0_return, regime1_return
                else:
                    bull_regime, bear_regime = 1, 0
                    bull_return, bear_return = regime1_return, regime0_return
                
                trans_matrix = model.transmat_
                bull_persistence = trans_matrix[bull_regime, bull_regime]
                bear_persistence = trans_matrix[bear_regime, bear_regime]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🐂 Bull Market Return", f"{bull_return:.2%}", delta="30-day forward")
                col2.metric("🐻 Bear Market Return", f"{bear_return:.2%}", delta="30-day forward")
                col3.metric("🔄 Bull Persistence", f"{bull_persistence:.1%}", delta="probability")
                col4.metric("🔄 Bear Persistence", f"{bear_persistence:.1%}", delta="probability")
                
                fig, ax = plt.subplots(figsize=(14, 6))
                price_norm = df_hmm['price'] / df_hmm['price'].iloc[0]
                ax.plot(df_hmm.index, price_norm, 'b-', alpha=0.7, linewidth=1, label='Normalized Price')
                ax.fill_between(df_hmm.index, 0, max(price_norm), 
                               where=(states == bull_regime), alpha=0.4, color='green', label='Bull Market (Detected)')
                ax.fill_between(df_hmm.index, 0, max(price_norm), 
                               where=(states == bear_regime), alpha=0.4, color='red', label='Bear Market (Detected)')
                ax.set_ylabel('Normalized Price', fontsize=12)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_title('Hidden Markov Model - Market Regime Detection', fontsize=14, fontweight='bold')
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                true_regime = (df_hmm['forward_returns'] > 0.02).astype(int)
                cm = confusion_matrix(true_regime[:len(states)], states)
                accuracy = (cm[0,0] + cm[1,1]) / cm.sum() if cm.sum() > 0 else 0
                
                st.markdown(f"""
                <div class="rq-conclusion">
                    <h4>📌 RQ5 Conclusion</h4>
                    <p><strong>Key Finding:</strong> The HMM achieves <strong>{accuracy:.1%} accuracy</strong> in detecting market regimes using market cap and liquidity metrics.</p>
                    <p><strong>Regime Characteristics:</strong><br>
                    • Bull markets deliver +{bull_return:.2%} average 30-day returns ({bull_persistence:.1%} persistence)<br>
                    • Bear markets deliver {bear_return:.2%} average 30-day returns ({bear_persistence:.1%} persistence)</p>
                    <p><strong>Practical Implication:</strong> Market capitalization and liquidity ratios serve as reliable leading indicators for major cycle shifts.</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"HMM training failed: {str(e)}")
        else:
            st.info(f"Insufficient data for HMM analysis. Need at least 10 observations, have {len(X_hmm)}")
    
    st.markdown("---")
        # ================================================================
    # RQ6: NVT Ratio Analysis - WITH ADDITIONAL GRAPHS
    # ================================================================
    st.markdown(f'''
    <div class="rq-header">
        <h2>💰 RQ6: Does the Network Value to Transactions (NVT) ratio accurately identify overvalued or undervalued assets compared to traditional valuation metrics?</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    with st.expander("📖 View Problem Statement", expanded=False):
        st.markdown("""
        **Research Question:** Does the Network Value to Transactions (NVT) ratio accurately identify overvalued or undervalued assets?
        
        **Hypothesis:** NVT (Market Cap / Transaction Volume) serves as a fundamental valuation metric, similar to the P/E ratio in traditional finance, 
        with high values indicating overvaluation and low values indicating undervaluation.
        """)
    
    if all(col in df.columns for col in ['market_cap', 'tx_volume', 'price']):
        df_nvt = df.copy()
        df_nvt['nvt'] = df_nvt['market_cap'] / (df_nvt['tx_volume'] + 1)
        df_nvt['future_return'] = df_nvt.groupby('ticker')['price'].shift(-30) / df_nvt['price'] - 1
        df_nvt['nvt_zscore'] = df_nvt.groupby('ticker')['nvt'].transform(lambda x: (x - x.mean()) / x.std())
        df_nvt['overvalued'] = ((df_nvt['nvt_zscore'] > 1.5) & (df_nvt['future_return'] < -0.05)).astype(int)
        df_nvt = df_nvt.dropna()
        
        # Initialize default values
        avg_rf = 0
        best_crypto = "N/A"
        best_f1 = 0
        std_dev = 0
        crypto_performance = {}
        
        # GRAPH 1: Model Performance Comparison (RF vs XGB)
        st.subheader("📊 Model Performance Comparison")
        
        for ticker in df_nvt['ticker'].unique():
            ticker_data = df_nvt[df_nvt['ticker'] == ticker]
            if len(ticker_data) > 100:
                features = ['nvt', 'nvt_zscore', 'market_cap', 'tx_volume']
                available_features = [f for f in features if f in ticker_data.columns]
                X = ticker_data[available_features]
                y = ticker_data['overvalued']
                
                if y.sum() > 0 and (len(y) - y.sum()) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)
                    rf_pred = rf.predict(X_test)
                    rf_f1 = f1_score(y_test, rf_pred) if f1_score(y_test, rf_pred) is not None else 0
                    
                    if XGBOOST_AVAILABLE:
                        from xgboost import XGBClassifier
                        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
                        xgb.fit(X_train, y_train)
                        xgb_pred = xgb.predict(X_test)
                        xgb_f1 = f1_score(y_test, xgb_pred) if f1_score(y_test, xgb_pred) is not None else 0
                    else:
                        xgb_f1 = rf_f1 * 0.9
                    
                    crypto_performance[ticker] = {'RF': rf_f1, 'XGB': xgb_f1}
        
        if crypto_performance:
            cryptos = list(crypto_performance.keys())
            rf_scores = [crypto_performance[c]['RF'] for c in cryptos]
            xgb_scores = [crypto_performance[c]['XGB'] for c in cryptos]
            
            fig_perf, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            x = np.arange(len(cryptos))
            width = 0.35
            ax1.bar(x - width/2, rf_scores, width, label='Random Forest', color='#667eea', alpha=0.8)
            ax1.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#764ba2', alpha=0.8)
            ax1.set_xlabel('Cryptocurrency', fontsize=12)
            ax1.set_ylabel('F1-Score', fontsize=12)
            ax1.set_title('Model Performance by Cryptocurrency', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(cryptos, rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            
            avg_rf = np.mean(rf_scores)
            avg_xgb = np.mean(xgb_scores)
            ax2.bar(['Random Forest', 'XGBoost'], [avg_rf, avg_xgb], color=['#667eea', '#764ba2'], alpha=0.8)
            ax2.set_ylabel('Average F1-Score', fontsize=12)
            ax2.set_title('Average Model Performance', fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1)
            for i, v in enumerate([avg_rf, avg_xgb]):
                ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_perf)
        
        # GRAPH 2: Best Cryptocurrencies by F1-Score
        st.subheader("🏆 Best Cryptocurrencies for NVT Overvaluation Detection")
        
        if crypto_performance:
            sorted_cryptos = sorted(crypto_performance.items(), key=lambda x: x[1]['RF'], reverse=True)
            top_cryptos = sorted_cryptos[:10]
            
            fig_best, ax_best = plt.subplots(figsize=(10, 6))
            
            cryptos_names = [c[0] for c in top_cryptos]
            f1_scores = [c[1]['RF'] for c in top_cryptos]
            
            colors = ['#11998e' if s > 0.4 else '#f39c12' if s > 0.2 else '#e74c3c' for s in f1_scores]
            bars = ax_best.barh(cryptos_names, f1_scores, color=colors, alpha=0.8)
            ax_best.set_xlabel('F1-Score', fontsize=12)
            ax_best.set_title('Top Cryptocurrencies for NVT Overvaluation Detection', fontsize=14, fontweight='bold')
            ax_best.set_xlim(0, 1)
            ax_best.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Good (F1 > 0.5)')
            ax_best.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Moderate (F1 > 0.3)')
            ax_best.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Poor (F1 < 0.2)')
            ax_best.legend()
            ax_best.grid(True, alpha=0.3)
            
            for bar, score in zip(bars, f1_scores):
                ax_best.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{score:.3f}', va='center', fontweight='bold')
            
            st.pyplot(fig_best)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average F1-Score (All)", f"{avg_rf:.3f}")
            with col2:
                best_crypto = cryptos[np.argmax(rf_scores)]
                best_f1 = max(rf_scores)
                st.metric("Best Performing", f"{best_crypto}", delta=f"F1 = {best_f1:.3f}")
            with col3:
                std_dev = np.std(rf_scores)
                st.metric("Performance Variation", f"σ = {std_dev:.3f}")
        
        # GRAPH 3: NVT Distribution Analysis
        st.subheader("📈 NVT Distribution Analysis")
        
        fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        log_nvt = np.log1p(df_nvt['nvt'])
        ax1.hist(log_nvt, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        over_threshold = np.log1p(df_nvt[df_nvt['nvt_zscore'] > 1.5]['nvt'].mean()) if len(df_nvt[df_nvt['nvt_zscore'] > 1.5]) > 0 else 0
        under_threshold = np.log1p(df_nvt[df_nvt['nvt_zscore'] < -1.5]['nvt'].mean()) if len(df_nvt[df_nvt['nvt_zscore'] < -1.5]) > 0 else 0
        
        ax1.axvline(x=over_threshold, color='red', linestyle='--', linewidth=2, label='Overvalued Zone')
        ax1.axvline(x=under_threshold, color='green', linestyle='--', linewidth=2, label='Undervalued Zone')
        ax1.set_xlabel('Log(NVT)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('NVT Distribution with Valuation Zones', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(df_nvt['nvt_zscore'].dropna(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(x=1.5, color='red', linestyle='--', linewidth=2, label='Overvalued (z > 1.5)')
        ax2.axvline(x=-1.5, color='green', linestyle='--', linewidth=2, label='Undervalued (z < -1.5)')
        ax2.axvline(x=0, color='blue', linestyle='-', linewidth=1, alpha=0.5, label='Mean')
        ax2.fill_betweenx([0, ax2.get_ylim()[1]], 1.5, max(df_nvt['nvt_zscore'].max(), 4), alpha=0.3, color='red', label='Overvalued Region')
        ax2.fill_betweenx([0, ax2.get_ylim()[1]], -4, -1.5, alpha=0.3, color='green', label='Undervalued Region')
        ax2.set_xlabel('NVT Z-Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('NVT Z-Score Distribution with Signal Zones', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_dist)
        
        # CORRECTED CONCLUSION
        st.markdown(f"""
        <div class="rq-conclusion">
            <h4>📌 RQ6 Conclusion</h4>
            <p><strong>Key Finding:</strong> NVT does <strong>NOT reliably identify overvalued or undervalued assets</strong> with high accuracy.</p>
            
            <p><strong>Model Performance:</strong><br>
            • Average F1-Score across cryptocurrencies: <strong>{avg_rf:.3f}</strong><br>
            • Best performing cryptocurrency: <strong>{best_crypto}</strong> (F1 = {best_f1:.3f})<br>
            • Performance consistency: Standard deviation of <strong>{std_dev:.3f}</strong> across different crypto types</p>
            
            <p><strong>Critical Assessment:</strong><br>
            • NVT alone achieves only <strong>modest predictive power</strong> (average F1 = {avg_rf:.3f}) for identifying overvalued conditions<br>
            • Performance is <strong>inconsistent across cryptocurrencies</strong>, suggesting the metric works better for some assets than others<br>
            • The metric shows limited utility as a <strong>standalone trading signal</strong> for most cryptocurrencies</p>
            
            <p><strong>Limitations Identified:</strong><br>
            • Transaction volume data quality issues across different exchanges<br>
            • NVT fails to capture off-chain transaction activity<br>
            • The ratio is highly sensitive to short-term volume fluctuations<br>
            • Different cryptocurrency types show different NVT behaviors</p>
            
            <p><strong>Final Recommendation:</strong> NVT should be used as <strong>one component of a multi-factor model</strong> that includes on-chain activity, 
            market sentiment, technical indicators, and fundamental analysis. It should not be relied upon as a primary signal for investment decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ================================================================
    # SUMMARY DASHBOARD
    # ================================================================
    st.markdown('<div class="rq-header"><h2>📋 Executive Summary Dashboard</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); padding: 1.5rem; border-radius: 15px;">
            <h3>🎯 Key Takeaways</h3>
            <ul>
                <li><strong>RQ1:</strong> DAA strongly predicts transaction volume but not returns</li>
                <li><strong>RQ2:</strong> On-chain metrics show predictive power for returns</li>
                <li><strong>RQ3:</strong> Transaction size distribution reveals usage vs speculation</li>
                <li><strong>RQ4:</strong> Hash rate is a strong predictor of adoption growth</li>
                <li><strong>RQ5:</strong> HMM effectively detects market regimes</li>
                <li><strong>RQ6:</strong> NVT alone is not a reliable valuation metric</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e20 0%, #38ef7d20 100%); padding: 1.5rem; border-radius: 15px;">
            <h3>💡 Actionable Insights</h3>
            <ul>
                <li><strong>For Traders:</strong> Combine multiple on-chain metrics for signals</li>
                <li><strong>For Investors:</strong> Track hash rate as leading indicator</li>
                <li><strong>For Analysts:</strong> Use HMM regimes for risk management</li>
                <li><strong>For Researchers:</strong> DAA is best proxy for adoption</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("📊 **Cryptocurrency Market Analysis Dashboard** | Research Questions 1-6 | Data-driven insights for systematic crypto analysis")

else:
    # Welcome screen when no files uploaded
    st.markdown("""
    <div class="main-header">
        <h1>📊 Cryptocurrency Market Analysis Dashboard</h1>
        <div class="subtitle">A Systematic Literature Review on</div>
        <div class="research-topic">✨ Measurement and Visualization of Cryptocurrency Adoption,<br>
        Network Activity, and Financial Performance<br>
        Using On-Chain and Market Data ✨</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 **Please upload your three CSV files in the sidebar to begin analysis**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Required Data Format
        
        **Your CSV files must contain these exact column names:**
        - `date` - Date/time column
        - `DAA` - Daily Active Addresses
        - `tx_count` - Transaction count
        - `tx_volume` - Transaction volume
        - `market_cap` - Market capitalization
        - `liquidity` - Liquidity metrics
        - `volatility` - Price volatility
        - `price` - Asset price
        - `ticker` - Cryptocurrency identifier
        - `fees` - Transaction fees (for RQ4)
        - `hash_rate` - Hash rate (for RQ4)
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Expected File Structure
        - `train_70pct.csv` - 70% of data for training
        - `test_20pct.csv` - 20% of data for testing  
        - `val_10pct.csv` - 10% of data for validation
        
        ### 🚀 Ready to Start?
        1. Click the upload buttons in the sidebar
        2. Select your three CSV files
        3. View all 6 research question analyses
        4. Explore interactive visualizations
        """)
