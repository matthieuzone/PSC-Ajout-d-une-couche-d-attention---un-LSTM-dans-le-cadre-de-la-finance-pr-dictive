import pandas as pd
import numpy as np
import math

##### Fourni par LUSIS #####

def daily_returns(trades):
    daily_pl = trades.groupby(pd.Grouper(freq='B'))['PL'].sum()
    daily_eq = trades.groupby(pd.Grouper(freq='B'))['Equity'].last()
    daily_ret = daily_pl / (daily_eq - daily_pl)
    return daily_ret.dropna()


def annualized_returns(returns, period=252):
    return np.cumprod(returns + 1)[-1] ** (period / len(returns)) - 1

def sortino_ratio(returns, target=0.0, period=252):
    tdd = np.sqrt(period * np.mean(np.minimum(0, returns - target / period) ** 2))
    return (annualized_returns(returns) - target) / tdd

def compute_mae_mfe(trades, bars, size=10000):
    mae = []
    mfe = []
    for index in range(trades.shape[0]):
        trade = trades[index:index + 1]
        if trade.EntryTime[0] == trade.ExitTime[0]:
            bar_from = 0
        else:
            bar_from = 1
        if trade.Side[0] == 'BUY':
            mae.append(size * (min(bars[trade.EntryTime[0]:trade.ExitTime[0]][bar_from:]['low']) - trade.EntryPrice[0]))
            mfe.append(size * (max(bars[trade.EntryTime[0]:trade.ExitTime[0]][bar_from:]['high']) - trade.EntryPrice[0]))
        else:
            mae.append(size * (trade.EntryPrice[0] - max(bars[trade.EntryTime[0]:trade.ExitTime[0]][bar_from:]['high'])))
            mfe.append(size * (trade.EntryPrice[0] - min(bars[trade.EntryTime[0]:trade.ExitTime[0]][bar_from:]['low'])))
    else:
        trades['MAE'] = mae
        trades['MFE'] = mfe
        return trades
    
    
def compute_sortino(bars, y_predict, y_datetime, entry_ref='close', exit_ref='close', predict_bars=1, bh=True, qty=10000, starting_equity=0, symbol=None, side='both', spread=0.0, sl=-np.inf, fixed_size=True, eq_mult=1, min_size=1000, name=None, author=None, note=None, showequitycurve=True, showtrades=True, showplstats=True, showmaemfe=True, printreport=True):
    predict_df = pd.DataFrame.from_dict(data={'date':y_datetime,  'predict':np.argmax(y_predict, axis=1) * 2 - 1}).set_index('date')
    trades = pd.merge(bars, predict_df, how='left', left_index=True, right_index=True)
    trades['EntryPrice'] = trades[entry_ref]
    trades['EntryTime'] = trades.index
    trades['ExitPrice'] = trades[exit_ref].shift(periods=(-1 * predict_bars))
    trades['ExitTime'] = trades['EntryTime'].shift(periods=(-1 * predict_bars))
    trades = trades.dropna()
    trades['Side'] = trades['predict'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
    if side == 'buy':
        trades = trades[trades['Side'] == 'BUY']
    else:
        if side == 'sell':
            trades = trades[trades['Side'] == 'SELL']
        else:
            trades = compute_mae_mfe(trades, bars, size=1)
            trades['PointPL'] = trades['predict'] * (trades['ExitPrice'] - trades['EntryPrice'])
            trades['PL'] = 0
            trades['Equity'] = 0
            if fixed_size:
                trades['TradeSize'] = qty
            else:
                trades['TradeSize'] = 0
        for index in range(trades.shape[0]):
            loc_pl = trades.columns.get_loc('PL')
            loc_eq = trades.columns.get_loc('Equity')
            loc_ts = trades.columns.get_loc('TradeSize')
            point_pl = trades.iloc[(index, trades.columns.get_loc('PointPL'))]
            mae = trades.iloc[(index, trades.columns.get_loc('MAE'))]
            mfe = trades.iloc[(index, trades.columns.get_loc('MFE'))]
            if sl > mae:
                point_pl = sl
                trades.iloc[(index, trades.columns.get_loc('PointPL'))] = point_pl
            elif index > 0:
                prev_equity = trades.iloc[(index - 1, loc_eq)]
            else:
                prev_equity = starting_equity
            if not fixed_size:
                trades.iloc[(index, loc_ts)] = math.floor(prev_equity * eq_mult / min_size) * min_size
            trades.iloc[(index, loc_pl)] = (point_pl - spread) * trades.iloc[(index, loc_ts)]
            trades.iloc[(index, loc_eq)] = prev_equity + trades.iloc[(index, loc_pl)]
        else:
            trades['CumPL'] = trades['PL'].cumsum()
            trades['Return'] = trades['PL'] / (trades['Equity'] - trades['PL'])
            trades['CumReturn'] = (trades['Return'] + 1).cumprod() - 1
            trades[['PL', 'Equity', 'CumPL']] = trades[['PL', 'Equity', 'CumPL']].round(2)
            trades['CumPL'] = trades['PL'].cumsum()
            trades['Return'] = trades['PL'] / (trades['Equity'] - trades['PL'])
            trades['CumReturn'] = (trades['Return'] + 1).cumprod() - 1
            returns = daily_returns(trades)
            sortino = sortino_ratio(returns)
            return sortino