import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# 設置應用標題
st.title('Fama-French五因子模型分析與Chow Test')

# 用戶輸入
ticker_symbol = st.text_input('請輸入股票代碼', 'AAPL')
start_date = st.date_input('整體分析開始日期')
end_date = st.date_input('整體分析結束日期')
start_period_1 = st.date_input('第一時間段開始日期')
end_period_1 = st.date_input('第一時間段結束日期')
start_period_2 = st.date_input('第二時間段開始日期')
end_period_2 = st.date_input('第二時間段結束日期')

# Fama-French 五因子模型數據路徑
file_path = r"F-F_Research_Data_5_Factors_2x3.csv"
# 讀取 Fama-French 五因子模型數據
factors_df = pd.read_csv(file_path, index_col='Date')
factors_df.index = pd.to_datetime(factors_df.index, format='%Y%m').strftime('%Y-%m-%d')
factors_df.index = pd.to_datetime(factors_df.index) + pd.offsets.MonthBegin(1)


# 定義執行分析的函數
def analyze_stock(ticker_symbol, start_date, end_date, start_period_1, end_period_1, start_period_2, end_period_2):
    # 下載股票數據
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data.index = pd.to_datetime(stock_data.index)

    # 計算月度回報率
    stock_monthly_returns = stock_data['Adj Close'].resample('MS').ffill().pct_change()
    
    # 準備合併的數據框架
    analysis_df = pd.DataFrame(index=stock_monthly_returns.index)
    analysis_df['Stock_Return'] = stock_monthly_returns
    
    # 合併股票回報率和五因子模型數據
    merged_df = analysis_df.join(factors_df).dropna()
    merged_df['Excess_Return'] = merged_df['Stock_Return'] - merged_df['RF']
    
    # 篩選時間段
    period1 = merged_df.loc[start_period_1:end_period_1]
    period2 = merged_df.loc[start_period_2:end_period_2]
    
    # 這裡可以添加回歸分析和Chow Test的代碼
    # ...
    import statsmodels.api as sm
    from scipy import stats

    # 定義一個用於執行回歸分析並返回模型結果的函數
    def perform_regression(data):
        # 因子作為自變量
        X = data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        # 超額回報率作為因變量
        y = data['Excess_Return']
        # 為自變量添加常數項
        X = sm.add_constant(X)
        # 執行OLS回歸分析
        model = sm.OLS(y, X).fit()
        return model
    
    # 對兩個時間段執行回歸分析
    model1 = perform_regression(period1)
    model2 = perform_regression(period2)

    # 展示回歸分析結果
    st.write('事件前回歸結果:')
    st.write(model1.summary())
    st.write('事件後回歸結果:')
    st.write(model2.summary())

    # Chow Test
    def chow_test(model1, model2, period1, period2):
        # 提取模型的殘差平方和
        rss1 = model1.ssr
        rss2 = model2.ssr
        rss_combined = sm.OLS(pd.concat([period1['Excess_Return'], period2['Excess_Return']]),
                              sm.add_constant(pd.concat([period1[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], 
                                                        period2[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]]))
                             ).fit().ssr
        # 計算自由度
        n = len(period1) + len(period2)
        k = len(model1.params)  # 或 model2.params，因為它們應該是相同的

        # 計算Chow統計量
        chow_statistic = ((rss_combined - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2*k))
        # 計算臨界值
        critical_value = stats.f.ppf(0.95, dfn=k, dfd=n-2*k)
        return chow_statistic, critical_value

    # 執行Chow Test
    chow_stat, critical_val = chow_test(model1, model2, period1, period2)

    # 判斷並展示Chow Test結果
    st.write('Chow Test統計量:', chow_stat)
    st.write('臨界值:', critical_val)
    if chow_stat > critical_val:
        st.write("存在顯著差異。(´・ω・)つ旦")
    else:
        st.write("沒有顯著差異。( ´•̥̥̥ω•̥̥̥` )")

    # 確保將結果顯示在 Streamlit 應用中，例如：
    # st.write(model.summary())

# 當用戶輸入股票代碼和時間段後，執行分析
if ticker_symbol and start_date and end_date and start_period_1 and end_period_1 and start_period_2 and end_period_2:
    analyze_stock(ticker_symbol, start_date, end_date, start_period_1, end_period_1, start_period_2, end_period_2)
