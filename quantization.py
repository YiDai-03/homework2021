#组合数学作业

import pandas as pd
import numpy as np
import datetime as dt
import talib as ta
from datetime import date,timedelta 
import statsmodels.api as sm
from jqlib.technical_analysis import *
from jqdata import *
from jqlib.optimizer import *

def TREND(security,date):
    EMA12 =  EXPMA(security,date,timeperiod = 12)[security]
    EMA50 =  EXPMA(security,date,timeperiod = 50)[security]
    level = 1
    #EMA
    if EMA12 > EMA50 : 
        level = 1
    else:
        level = 0 
    R=[1.5,3.8] 
    C=[99.5,98] 
    return R[level],C[level],level

def VaR_posit(security,risk_exposure,CI_alpha): 
    risk_exposure = risk_exposure
    CI_alpha = CI_alpha 
    risk_money = -1* risk_exposure
    price  = attribute_history(security,125,'1d',['close'])
    chage = price.pct_change().dropna(axis=0)
    VaR = np.nanpercentile(chage,(100-CI_alpha))
    target_value = risk_money/VaR
    target_positions = round(target_value/1,6)
    if target_positions >= 100: target_positions = 100
    return target_positions
    
def market_open(context):

    if g.change >= 0:
        
        change_value = g.change * context.portfolio.total_value
        order_target_value(g.security, change_value)
        
        g.position = context.portfolio.positions_value/context.portfolio.total_value
    

def after_market_close(context):
    print('当前仓位：%.2f'%g.position)
    record(posit=round(g.position,2))
        
    


def initialize(context):
    set_params(context)
    g.position = 0.0 
    g.change = 0.0 
    set_backtest()

    run_daily(before_market_open, time='before_open')
    run_daily(stop_loss, time='09:30', reference_security='000300.XSHG')
    run_daily(market_open, time='open')
    run_daily(after_market_close, time='after_close')
def before_market_open(context):
    g.security = '000300.XSHG'
    R = 0
    C = 0
    R,C,Level = TREND(g.security,context.previous_date)
    g.change = VaR_posit(g.security,R,C)/100
    print Level,g.change

def set_params(context):
    g.tc = 20 * 3                          
    g.t = 0
    g.big_small = 'big'                   
    context.stock = '000300.XSHG'
    # g.num = 5
    g.radio = 0.003
    g.stock='000300.XSHG'                 
    g.buy_list = []
    g.sell_list = []
    g.stock_wight = {}

    
def set_backtest():
    set_benchmark('000300.XSHG')               
    set_slippage(FixedSlippage(0.02))      


def before_trading(context):
    pass



def stop_loss(context):
    g.buy_list = []
    g.sell_list = []
    stock_list = context.portfolio.positions.keys()
    if len(stock_list) == 0:
        return

    current_data = get_current_data()  
    last_date = context.current_dt  
  
    for stock in stock_list:
        cumulative_return=current_data[stock].day_open/context.portfolio.positions[stock].avg_cost
        if cumulative_return < 0.9:
            order_target_value(stock,0)
            g.sell_list.append(stock)
          
 
    
    
    # 2. 根据peg止损
    peg = get_peg(stock_list, last_date).T

    for key in peg.keys(): 

        if peg[key] > 0.5 or peg[key] < 0:
            order_target_value(key, 0)
            g.sell_list.append(key)
            log.info('peg 卖出\n', key)


def handle_data(context, data):
    if  g.t % g.tc == 0:
        last_date = context.current_dt  
        stock_list=list(get_all_securities(['stock'],date=last_date).index)
        current_data = get_current_data()  

        stock_list=fun_unpaused(current_data, stock_list)
        stock_list=fun_st(current_data, stock_list)
        stock_list=fun_highlimit(current_data, stock_list)
        stock_list=fun_remove_new(context, stock_list, 60)
        
       
        concat_obj = []
        # 以下是财务指标因子
        f_alpha_list = get_fundamentals_alpha(stock_list, last_date)
        concat_obj.extend(f_alpha_list)
        
        df = pd.concat(concat_obj, axis=1)
        df = df.dropna()
        log.info("最后因子:\n", df.head(5))
        sum = df.sum(axis=1)


        if g.big_small == 'big':
            sum.order(ascending = False,inplace=True)
        if g.big_small == 'small':
            sum.order(ascending = True,inplace=True)


        stock_list1 = sum[0:int(len(stock_list) * g.radio)].index

        buy_list = []
        for stock in stock_list1:
            buy_list.append(stock)
        
        sell_list = set(context.portfolio.positions.keys()) - set(buy_list)

        for stock in sell_list:
            order_target_value(stock, 0)
            g.sell_list.append(stock)
        

        optimized_weight = portfolio_optimizer(date=context.previous_date,
                                    securities = buy_list,
                                    target = MinVariance(count=250),
                                    constraints = [WeightConstraint(low=g.position-0.05, high=g.position+0.05),
                                                  AnnualProfitConstraint(limit=0.1, count=250)],
                                    bounds=[],
                                    default_port_weight_range=[0., 1.0],
                                    ftol=1e-09,
                                    return_none_if_fail=True)        
        

        log.info("optimized_weight\n", optimized_weight)
        cash = context.portfolio.portfolio_value

        if type(optimized_weight) == type(None):

            for stock in buy_list:
                order_target_value(stock,cash / int(len(stock_list) * g.radio))
                g.buy_list.append(stock)

        else:
            g.stock_wight = optimized_weight
            for stock in optimized_weight.keys():
                value = cash * optimized_weight[stock] # 确定每个标的的权重
                order_target_value(stock, value) 
                g.buy_list.append(stock)
        
        
    g.t=g.t+1
    send_message_to_WeChat()

def send_message_to_WeChat():
    buy_list_str = []
    sell_list_str = []
    weight_list_str = []
    for code in g.buy_list : 
        buy_list_str.append("\t".join([code , get_security_info(code).display_name]))
    for code in g.sell_list : 
        sell_list_str.append("\t".join([code , get_security_info(code).display_name]))
    for code in g.stock_wight.keys():
        weight = str(format(g.stock_wight[code], '.2f'))
        weight_list_str.append("\t".join([code,  weight]))
    buy_result = "\n==========买入股票==========\n" + "\n".join(buy_list_str)
    sell_result = "\n==========卖出股票==========\n" + "\n".join(sell_list_str)
    weight_str = "\n==========各股票权重==========\n" + "\n".join(weight_list_str)
    index_days = "\n==========当前天数==========\n" + str(g.t % g.tc)
    adjust_positions_days = "\n==========调仓天数==========\n" + str(g.tc)
    message_str = "\n".join([buy_result, sell_result, weight_str,index_days, adjust_positions_days])
    send_message(message_str)
    log.info(message_str)


def get_fundamentals_alpha(stock_list, last_date):
    alpha_name_direction = {
        # "market_cap":-1,
        # "pe_ratio":-1,
        # "pb_ratio":-1,
        # "ps_ratio":-1,
        # "financing_expense_to_total_revenue":-1,
        # "roe":1,
        # "inc_net_profit_year_on_year":1,
        # "net_profit_to_total_revenue":1,
    }
    df = get_fundamentals(
        query(
         valuation.code, 
         valuation.market_cap,
         valuation.pe_ratio,
         valuation.pb_ratio,
         valuation.ps_ratio,
         indicator.financing_expense_to_total_revenue,
         indicator.roe,
         indicator.inc_net_profit_year_on_year,
         indicator.net_profit_to_total_revenue,
    ).filter(
        valuation.code.in_(stock_list),
        valuation.pe_ratio > 0,
        indicator.inc_net_profit_year_on_year > 0,)
        ,date=last_date)
    df = df.set_index('code')
    alpha_list = []
    for key, value in alpha_name_direction.items():
        after_MAD = MAD(key, df) # 绝对中位数法取极值
        after_zscore = zscore(key, after_MAD) # z-score法标准化
        alpha = after_zscore * value # 取方向
        alpha_list.append(alpha[key])
    # 彼得林奇PEG选股因子 市盈率(PE_ttm)和单季度的净利润增长率(growth_rate) PEG = PE_ttm / growth_rate  PEG越接近0(越小)，说明越被低估
    df['peg'] = df['pe_ratio'] / df['inc_net_profit_year_on_year'] 
    log.info('df\n', df.head(5))
    after_MAD = MAD('peg', df) # 绝对中位数法取极值
    after_zscore = zscore('peg', after_MAD) # z-score法标准化
    alpha = after_zscore * -1 # 取方向
    alpha_list.append(alpha['peg'])
    return alpha_list


def get_peg(stock_list, last_date):
    df = get_fundamentals(
        query(
         valuation.code, 
         valuation.pe_ratio,
         indicator.inc_net_profit_year_on_year,
    ).filter(
        valuation.code.in_(stock_list),
        valuation.pe_ratio > 0,
        indicator.inc_net_profit_year_on_year > 0,
        
        ),date=last_date)
    df = df.set_index('code')
    df['peg'] = df['pe_ratio'] / df['inc_net_profit_year_on_year'] 
    return df['peg']
    
def get_index_pe(last_date):
    stock_list=list(get_all_securities(['stock'],date=last_date).index)
    df = get_fundamentals(
         query(
             valuation.code, 
             valuation.market_cap,
             valuation.pe_ratio,
         ).filter(valuation.code.in_(stock_list))
         ,date=last_date)
    df = df.set_index('code')
    df = df['pe_ratio'].order(ascending = True)
    pe_ratio = df[df > 0]
    return 1/np.mean(pe_ratio[0:int(len(pe_ratio)/4)])

"""
以下是进行因子数据处理，对因子进行MAD去极值，以及标准化处理
"""    
def MAD(factor, df):
    # 取得中位数
    median = df[factor].median()
    # 取得数据与中位数差值
    df1 = df-median
    # 取得差值绝对值
    df1 = df1.abs()
    # 取得绝对中位数
    MAD = df1[factor].median()
    # 得到数据上下边界
    extreme_upper = median + 3 * 1.483 * MAD
    extreme_lower = median - 3 * 1.483 * MAD
    # 将数据上下边界外的数值归到边界上
    df.ix[(df[factor]<extreme_lower), factor] = extreme_lower
    df.ix[(df[factor]>extreme_upper), factor] = extreme_upper
    return df



# z-score标准化
def zscore(factor, df):
    # 取得均值
    mean = df[factor].mean()
    # 取得标准差
    std = df[factor].std()
    # 取得标准化后数据
    df = (df - mean) / std
    return df    


"""
以下对股票列表进行去除ST，停牌，去新股，以及去除开盘涨停股
"""   
#去除开盘涨停股票
def fun_highlimit(current_data,_stock_list):
    stock_list = []
    for stock in _stock_list:
        try:
            if current_data[stock].day_open!=current_data[stock].high_limit :
                stock_list.append(stock)
        except Exception as e:
            raise e
    return stock_list

#去除st股票
def fun_st(current_data,_stock_list): 
    stock_list = []
    for stock in _stock_list:
        try:
            if not current_data[stock].is_st :
                stock_list.append(stock)
        except Exception as e:
            raise e
    return stock_list

def fun_unpaused(current_data, _stock_list):
    stock_list = []
    for stock in _stock_list:
        try:
            if not current_data[stock].paused :
                stock_list.append(stock)
        except Exception as e:
            raise e
    return stock_list    


def fun_remove_new(context, _stock_list, days):
    deltaDate =  context.current_dt - dt.timedelta(days)
    # log.info("my" , deltaDate)
    stock_list = []
    for stock in _stock_list:
        # log.info("my" , get_security_info(stock).start_date)
        start_date = str(get_security_info(stock).start_date)
        if dt.datetime.strptime(start_date, '%Y-%m-%d') < deltaDate:
            stock_list.append(stock)
    return stock_list        

