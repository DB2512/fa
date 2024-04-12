import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# 设置应用标题
st.title('Fama-French五因子模型分析与Chow Test')

# 用户输入
ticker_symbol = st.text_input('请输入股票代码', 'AAPL')
start_date = st.date_input('整体分析开始日期')
end_date = st.date_input('整体分析结束日期')
start_period_1 = st.date_input('第一时间段开始日期')
end_period_1 = st.date_input('第一时间段结束日期')
start_period_2 = st.date_input('第二时间段开始日期')
end_period_2 = st.date_input('第二时间段结束日期')

# Fama-French 五因子模型数据路径
file_path = "data/F-F_Research_Data_5_Factors_2x3.csv"
# 读取 Fama-French 五因子模型数据
factors_df = pd.read_csv(file_path, index_col='Date')
factors_df.index = pd.to_datetime(factors_df.index, format='%Y%m').strftime('%Y-%m-%d')
factors_df.index = pd.to_datetime(factors_df.index) + pd.offsets.MonthBegin(1)


# 定义执行分析的函数
def analyze_stock(ticker_symbol, start_date, end_date, start_period_1, end_period_1, start_period_2, end_period_2):
    # 下载股票数据
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    stock_data.index = pd.to_datetime(stock_data.index)

    # 计算月度回报率
    stock_monthly_returns = stock_data['Adj Close'].resample('MS').ffill().pct_change()
    
    # 准备合并的数据框架
    analysis_df = pd.DataFrame(index=stock_monthly_returns.index)
    analysis_df['Stock_Return'] = stock_monthly_returns
    
    # 合并股票回报率和五因子模型数据
    merged_df = analysis_df.join(factors_df).dropna()
    merged_df['Excess_Return'] = merged_df['Stock_Return'] - merged_df['RF']
    
    # 筛选时间段
    period1 = merged_df.loc[start_period_1:end_period_1]
    period2 = merged_df.loc[start_period_2:end_period_2]
    
    # 这里可以添加回归分析和Chow Test的代码
    # ...
    import statsmodels.api as sm
    from scipy import stats

    # 定义一个用于执行回归分析并返回模型结果的函数
    def perform_regression(data):
        # 因子作为自变量
        X = data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        # 超额回报率作为因变量
        y = data['Excess_Return']
        # 为自变量添加常数项
        X = sm.add_constant(X)
        # 执行OLS回归分析
        model = sm.OLS(y, X).fit()
        return model
    
    # 对两个时间段执行回归分析
    model1 = perform_regression(period1)
    model2 = perform_regression(period2)

    # 展示回归分析结果
    st.write('第一时间段回归结果:')
    st.write(model1.summary())
    st.write('第二时间段回归结果:')
    st.write(model2.summary())

    # Chow Test
    def chow_test(model1, model2, period1, period2):
        # 提取模型的残差平方和
        rss1 = model1.ssr
        rss2 = model2.ssr
        rss_combined = sm.OLS(pd.concat([period1['Excess_Return'], period2['Excess_Return']]),
                              sm.add_constant(pd.concat([period1[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']], 
                                                        period2[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]]))
                             ).fit().ssr
        # 计算自由度
        n = len(period1) + len(period2)
        k = len(model1.params)  # 或 model2.params，因为它们应该是相同的
        # 计算Chow统计量
        chow_statistic = ((rss_combined - (rss1 + rss2)) / k) / ((rss1 + rss2) / (n - 2*k))
        # 计算临界值
        critical_value = stats.f.ppf(0.95, dfn=k, dfd=n-2*k)
        return chow_statistic, critical_value

    # 执行Chow Test
    chow_stat, critical_val = chow_test(model1, model2, period1, period2)

    # 判断并展示Chow Test结果
    st.write('Chow Test统计量:', chow_stat)
    st.write('临界值:', critical_val)
    if chow_stat > critical_val:
        st.write("在95%的置信水平下拒绝原假设，两时间段的回归系数存在显著差异。")
    else:
        st.write("在95%的置信水平下无法拒绝原假设，两时间段的回归系数没有显著差异。")

    # 确保将结果显示在 Streamlit 应用中，例如：
    # st.write(model.summary())

# 当用户输入股票代码和时间段后，执行分析
if ticker_symbol and start_date and end_date and start_period_1 and end_period_1 and start_period_2 and end_period_2:
    analyze_stock(ticker_symbol, start_date, end_date, start_period_1, end_period_1, start_period_2, end_period_2)
