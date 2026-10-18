import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import timedelta
import time

st.set_page_config(page_title='Stock Analytics Dashboard', page_icon='📈', layout='wide')
TRADING_DAYS = 252
RISK_FREE_RATE = 0.04
DEFAULT_DATASET = 'comprehensive_stock_dataset.csv'


def load_css():
    st.markdown('''<style>
    .main {background-color:#0e1117;} .block-container{padding-top:1.5rem;}
    .hero{padding:1.4rem;border-radius:16px;background:linear-gradient(90deg,#25104a,#063a47);text-align:center;margin-bottom:1rem}
    .hero h1{margin:0}.small-note{opacity:.75;font-size:.9rem}
    </style>''', unsafe_allow_html=True)


def prepare_data(df):
    df = df.copy()
    required = ['Date','Ticker','Sector','Open','High','Low','Close','Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for c in ['Open','High','Low','Close','Volume','Adj Close','Market_Cap','Daily_Return','Price_Change']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Date','Ticker','Close']).sort_values(['Ticker','Date'])
    if 'Adj Close' not in df: df['Adj Close'] = df['Close']
    if 'Daily_Return' not in df: df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
    else: df['Daily_Return'] = df['Daily_Return'].fillna(df.groupby('Ticker')['Close'].pct_change())
    if 'Price_Change' not in df: df['Price_Change'] = df.groupby('Ticker')['Close'].diff()
    if 'Market_Cap' not in df: df['Market_Cap'] = np.nan
    return df


@st.cache_data
def load_default_data(path):
    return prepare_data(pd.read_csv(path))


def add_indicators(d):
    d = d.sort_values('Date').copy()
    d['Return'] = d['Close'].pct_change()
    d['Cumulative_Return'] = (1 + d['Return'].fillna(0)).cumprod() - 1
    d['SMA20'] = d['Close'].rolling(20).mean(); d['SMA50'] = d['Close'].rolling(50).mean(); d['SMA200'] = d['Close'].rolling(200).mean()
    d['EMA12'] = d['Close'].ewm(span=12, adjust=False).mean(); d['EMA26'] = d['Close'].ewm(span=26, adjust=False).mean()
    d['MACD'] = d['EMA12'] - d['EMA26']; d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean(); d['MACD_Hist'] = d['MACD'] - d['MACD_Signal']
    delta = d['Close'].diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan); d['RSI'] = 100 - 100/(1+rs)
    d['BB_Middle'] = d['Close'].rolling(20).mean(); sd = d['Close'].rolling(20).std(); d['BB_Upper'] = d['BB_Middle'] + 2*sd; d['BB_Lower'] = d['BB_Middle'] - 2*sd
    d['Rolling_Vol_20'] = d['Return'].rolling(20).std() * np.sqrt(TRADING_DAYS) * 100
    d['Running_Peak'] = d['Close'].cummax(); d['Drawdown'] = d['Close']/d['Running_Peak'] - 1
    mean = d['Return'].mean(); std = d['Return'].std(); d['Return_Z'] = (d['Return']-mean)/std if std and not np.isnan(std) else 0
    d['Anomaly'] = d['Return_Z'].abs() > 3
    d['Rolling_Sharpe_60'] = ((d['Return'].rolling(60).mean()*TRADING_DAYS - RISK_FREE_RATE) / (d['Return'].rolling(60).std()*np.sqrt(TRADING_DAYS)))
    d['Vol_MA20'] = d['Volume'].rolling(20).mean()
    return d


def risk_metrics(d, market=None):
    r = d['Return'].dropna()
    if r.empty: return {k:np.nan for k in ['Daily Return','Annualized Volatility','Max Drawdown','Sharpe Ratio','Sortino Ratio','VaR 95%','Beta']}
    ann_ret = r.mean()*TRADING_DAYS; ann_vol = r.std()*np.sqrt(TRADING_DAYS)
    downside = r[r<0].std()*np.sqrt(TRADING_DAYS)
    beta = np.nan
    if market is not None:
        a = d[['Date','Return']].dropna().merge(market[['Date','Return']].dropna(), on='Date', suffixes=('_stock','_market'))
        if len(a)>2 and a['Return_market'].var()!=0: beta = a['Return_stock'].cov(a['Return_market'])/a['Return_market'].var()
    return {
        'Daily Return': r.iloc[-1]*100,
        'Annualized Volatility': ann_vol*100,
        'Max Drawdown': d['Drawdown'].min()*100,
        'Sharpe Ratio': (ann_ret-RISK_FREE_RATE)/ann_vol if ann_vol else np.nan,
        'Sortino Ratio': (ann_ret-RISK_FREE_RATE)/downside if downside else np.nan,
        'VaR 95%': np.percentile(r,5)*100,
        'Beta': beta
    }


def fmt(v, suffix='', digits=2): return 'N/A' if pd.isna(v) else f'{v:,.{digits}f}{suffix}'


def kpis(d, market):
    r=d['Return'].dropna(); current=d['Close'].iloc[-1]; first=d['Close'].iloc[0]
    total=(current/first-1)*100; ann=((current/first)**(TRADING_DAYS/max(len(d)-1,1))-1)*100
    risk=risk_metrics(d,market); high52=d.tail(min(252,len(d)))['High'].max(); low52=d.tail(min(252,len(d)))['Low'].min()
    values=[
      ('Current Price',fmt(current,'',2)),('Total Return',fmt(total,'%',2)),('Daily Return',fmt(risk['Daily Return'],'%',2)),('Annualized Return',fmt(ann,'%',2)),
      ('Annual Volatility',fmt(risk['Annualized Volatility'],'%',2)),('Max Drawdown',fmt(risk['Max Drawdown'],'%',2)),('Sharpe Ratio',fmt(risk['Sharpe Ratio'],'',2)),('Sortino Ratio',fmt(risk['Sortino Ratio'],'',2)),
      ('VaR (95%, 1-Day)',fmt(risk['VaR 95%'],'%',2)),('Beta vs Market',fmt(risk['Beta'],'',2)),('RSI (14)',fmt(d['RSI'].iloc[-1],'',1)),('MACD',fmt(d['MACD'].iloc[-1],'',2)),
      ('SMA 20',fmt(d['SMA20'].iloc[-1],'',2)),('SMA 50',fmt(d['SMA50'].iloc[-1],'',2)),('52W High',fmt(high52,'',2)),('52W Low',fmt(low52,'',2)),
      ('Avg Volume',fmt(d['Volume'].mean(),'',0)),('Skewness',fmt(r.skew(),'',2)),('Kurtosis',fmt(r.kurtosis(),'',2)),('Anomalies',str(int(d['Anomaly'].sum())))
    ]
    st.caption('20 KPIs covering performance, technical signals, liquidity, statistics and risk.')
    for start in range(0,len(values),5):
        cols=st.columns(5)
        for col,(name,val) in zip(cols,values[start:start+5]): col.metric(name,val)


def chart(title, fig):
    fig.update_layout(title=title, template='plotly_dark', height=390, margin=dict(l=20,r=20,t=55,b=20), legend_orientation='h')
    st.plotly_chart(fig, use_container_width=True)


def show_visuals(d, df, selected, start, end, market):
    tabs=st.tabs(['Price & Trend','Technical','Risk & Returns','Volume & Statistics','Comparison','Portfolio'])
    with tabs[0]:
        f=go.Figure(go.Candlestick(x=d.Date,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='OHLC')); chart('1. Candlestick Price',f)
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.Close,name='Close'); f.add_scatter(x=d.Date,y=d.SMA20,name='SMA20'); f.add_scatter(x=d.Date,y=d.SMA50,name='SMA50'); chart('2. Price with SMA20 & SMA50',f)
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.Close,name='Close'); f.add_scatter(x=d.Date,y=d.EMA12,name='EMA12'); f.add_scatter(x=d.Date,y=d.EMA26,name='EMA26'); chart('3. EMA Trend',f)
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.Close,name='Close'); f.add_scatter(x=d.Date,y=d.BB_Upper,name='Upper'); f.add_scatter(x=d.Date,y=d.BB_Middle,name='Middle'); f.add_scatter(x=d.Date,y=d.BB_Lower,name='Lower'); chart('4. Bollinger Bands',f)
    with tabs[1]:
        f=go.Figure(go.Scatter(x=d.Date,y=d.RSI,name='RSI')); f.add_hline(y=70,line_dash='dash'); f.add_hline(y=30,line_dash='dash'); chart('5. RSI (14)',f)
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.MACD,name='MACD'); f.add_scatter(x=d.Date,y=d.MACD_Signal,name='Signal'); chart('6. MACD & Signal',f)
        chart('7. MACD Histogram',go.Figure(go.Bar(x=d.Date,y=d.MACD_Hist,name='Histogram')))
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.SMA20,name='SMA20'); f.add_scatter(x=d.Date,y=d.SMA50,name='SMA50'); f.add_scatter(x=d.Date,y=d.SMA200,name='SMA200'); chart('8. Moving Average Structure',f)
    with tabs[2]:
        chart('9. Daily Returns',go.Figure(go.Scatter(x=d.Date,y=d.Return*100,name='Daily Return %')))
        chart('10. Cumulative Returns',go.Figure(go.Scatter(x=d.Date,y=d.Cumulative_Return*100,name='Cumulative Return %')))
        chart('11. 20-Day Rolling Annualized Volatility',go.Figure(go.Scatter(x=d.Date,y=d.Rolling_Vol_20,name='Volatility %')))
        chart('12. Drawdown',go.Figure(go.Scatter(x=d.Date,y=d.Drawdown*100,fill='tozeroy',name='Drawdown %')))
        chart('13. Rolling Sharpe Ratio (60D)',go.Figure(go.Scatter(x=d.Date,y=d.Rolling_Sharpe_60,name='Rolling Sharpe')))
        hist=px.histogram(d.dropna(subset=['Return']),x='Return',nbins=50); chart('14. Return Distribution',hist)
        an=d[d.Anomaly]; f=go.Figure(); f.add_scatter(x=d.Date,y=d.Return*100,name='Return %'); f.add_scatter(x=an.Date,y=an.Return*100,mode='markers',marker=dict(size=10,symbol='x'),name='|Z| > 3'); chart('15. Return Anomalies (Z-Score)',f)
    with tabs[3]:
        chart('16. Trading Volume',go.Figure(go.Bar(x=d.Date,y=d.Volume,name='Volume')))
        f=go.Figure(); f.add_scatter(x=d.Date,y=d.Volume,name='Volume'); f.add_scatter(x=d.Date,y=d.Vol_MA20,name='20D Avg Volume'); chart('17. Volume vs 20-Day Average',f)
        chart('18. Price vs Volume',px.scatter(d,x='Volume',y='Close',hover_data=['Date']))
        monthly=d.set_index('Date')['Return'].resample('ME').apply(lambda x:(1+x.dropna()).prod()-1).reset_index(); chart('19. Monthly Returns',px.bar(monthly,x='Date',y='Return'))
    with tabs[4]:
        opts=[x for x in sorted(df.Ticker.unique()) if x not in [selected,'MARKET_INDEX']]
        comps=st.multiselect('Compare with up to 3 stocks',opts,max_selections=3,key='compare')
        names=[selected]+comps
        cd=df[(df.Ticker.isin(names))&(df.Date>=pd.to_datetime(start))&(df.Date<=pd.to_datetime(end))]
        pivot=cd.pivot(index='Date',columns='Ticker',values='Close').dropna(how='all')
        if not pivot.empty:
            norm=pivot/pivot.iloc[0]*100; chart('20. Normalized Multi-Stock Performance',px.line(norm.reset_index(),x='Date',y=list(norm.columns)))
        stats=[]
        for t in names:
            x=add_indicators(cd[cd.Ticker==t]); rr=x.Return.dropna(); stats.append({'Ticker':t,'Annual Return %':rr.mean()*TRADING_DAYS*100,'Annual Volatility %':rr.std()*np.sqrt(TRADING_DAYS)*100})
        chart('21. Risk-Return Comparison',px.scatter(pd.DataFrame(stats),x='Annual Volatility %',y='Annual Return %',text='Ticker'))
        ret=cd.pivot(index='Date',columns='Ticker',values='Close').pct_change().dropna();
        if ret.shape[1]>1: chart('22. Return Correlation Heatmap',px.imshow(ret.corr(),text_auto='.2f',aspect='auto'))
    with tabs[5]:
        portfolio_ui(df,start,end,market)


def portfolio_ui(df,start,end,market):
    universe=[x for x in sorted(df.Ticker.unique()) if x!='MARKET_INDEX']
    chosen=st.multiselect('Portfolio stocks',universe,default=universe[:3],max_selections=5,key='portfolio_stocks')
    if not chosen: st.info('Select at least one stock.'); return
    weights=[]; cols=st.columns(len(chosen))
    for i,(c,t) in enumerate(zip(cols,chosen)): weights.append(c.number_input(f'{t} weight %',0.0,100.0,round(100/len(chosen),1),key='w_'+t))
    w=np.array(weights,dtype=float)
    if w.sum()==0: st.warning('Weights must total more than 0.'); return
    if abs(w.sum()-100)>0.1: st.info(f'Weights total {w.sum():.1f}%; normalized to 100% for calculation.')
    w=w/w.sum(); x=df[(df.Ticker.isin(chosen))&(df.Date>=pd.to_datetime(start))&(df.Date<=pd.to_datetime(end))]
    p=x.pivot(index='Date',columns='Ticker',values='Close')[chosen].dropna(); rets=p.pct_change().dropna(); pr=rets.dot(w)
    cumulative=(1+pr).cumprod(); draw=cumulative/cumulative.cummax()-1; ann_ret=pr.mean()*TRADING_DAYS; vol=pr.std()*np.sqrt(TRADING_DAYS); downside=pr[pr<0].std()*np.sqrt(TRADING_DAYS)
    vals=[('Portfolio Return',f'{(cumulative.iloc[-1]-1)*100:.2f}%'),('Portfolio Volatility',f'{vol*100:.2f}%'),('Portfolio Max Drawdown',f'{draw.min()*100:.2f}%'),('Portfolio Sharpe',f'{((ann_ret-RISK_FREE_RATE)/vol):.2f}' if vol else 'N/A'),('Portfolio Sortino',f'{((ann_ret-RISK_FREE_RATE)/downside):.2f}' if downside else 'N/A')]
    for c,(n,v) in zip(st.columns(5),vals): c.metric(n,v)
    chart('23. Portfolio Cumulative Performance',go.Figure(go.Scatter(x=cumulative.index,y=(cumulative-1)*100,name='Portfolio %')))
    chart('24. Portfolio Drawdown',go.Figure(go.Scatter(x=draw.index,y=draw*100,fill='tozeroy',name='Drawdown %')))
    contrib=pd.DataFrame({'Ticker':chosen,'Weight %':w*100,'Annual Return %':[rets[t].mean()*TRADING_DAYS*100 for t in chosen]}); chart('25. Portfolio Allocation',px.pie(contrib,names='Ticker',values='Weight %'))


def main():
    load_css(); st.markdown("<div class='hero'><h1>📈 Stock Market Analysis Dashboard</h1><div>Performance • Technical Analysis • Risk • Anomalies • Portfolio • Comparison</div></div>",unsafe_allow_html=True)
    with st.sidebar:
        st.header('Data & Controls')
        uploaded=st.file_uploader('Upload stock CSV',type=['csv'])
        path=Path(DEFAULT_DATASET)
        try:
            df=prepare_data(pd.read_csv(uploaded)) if uploaded else load_default_data(str(path))
        except Exception as e:
            st.error(f'Dataset could not be loaded: {e}'); st.stop()
        tradable=[x for x in sorted(df.Ticker.unique()) if x!='MARKET_INDEX'] or sorted(df.Ticker.unique())
        selected=st.selectbox('Stock',tradable)
        start=st.date_input('Start date',df.Date.min().date(),min_value=df.Date.min().date(),max_value=df.Date.max().date())
        end=st.date_input('End date',df.Date.max().date(),min_value=df.Date.min().date(),max_value=df.Date.max().date())
        refresh=st.toggle('Auto-refresh dashboard',False)
        refresh_seconds=st.selectbox('Refresh interval (seconds)',[30,60,120],index=0,disabled=not refresh)
        st.caption('Auto-refresh re-reads/recomputes the latest available dataset. It is refresh-capable, not a live exchange feed.')
    if start>end: st.error('Start date must be before end date.'); st.stop()
    filtered=df[(df.Ticker==selected)&(df.Date>=pd.to_datetime(start))&(df.Date<=pd.to_datetime(end))]
    if len(filtered)<2: st.warning('Not enough rows for analysis.'); st.stop()
    d=add_indicators(filtered)
    market_raw=df[(df.Ticker=='MARKET_INDEX')&(df.Date>=pd.to_datetime(start))&(df.Date<=pd.to_datetime(end))]
    market=add_indicators(market_raw) if len(market_raw)>1 else None
    st.subheader(f'{selected} — KPI Overview'); kpis(d,market)
    st.divider(); show_visuals(d,df,selected,start,end,market)
    st.divider(); st.caption(f'Rows in selected analysis: {len(d):,} | Data through {d.Date.max().date()} | Anomaly rule: |return Z-score| > 3')
    if refresh:
        time.sleep(refresh_seconds); st.rerun()

if __name__=='__main__': main()
