import aiutils.aiutils as ai
import aiutils.features_manager as feat
import aiutils.labels_manager as lab
import pandas_ta as ta

### Fourni par LUSIS sauf les features EMA et Stochastiques ###

pair = 'EURUSD'
df = ai.read_timeseries_from_csv(filename=f'./{pair}_1H_UTC.csv',closedatetimeref=False,totimescale='1H',fromtimezone='UTC',totimezone='Europe/Paris')
# print((df['close'].shift(-1) - df['close']))
lab.add_close_close(df,nbars=8,label=(0,1))
lab.add_close_close(df,nbars=16,label=(0,1))
df['range'] = df['high'] - df['low']

for index in range(1,4):
    feat.add_ohlc_features(df, f'C{index}O{index}')
    feat.add_ohlc_features(df, f'C{index}O{index-1}')

df.ta.rsi(append=True)
df[f'RSI_{14}']=df[f'RSI_{14}'].shift(1)
df['MA20'] = df['close'].shift(1).rolling(window=20).mean()
df['ecart'] = (df['close'] - df['MA20'] )
df = df.dropna()

def create_features(data, adx = True,use_bol = True,macd = True,diff_ll = True,rol_means = True,kama = True,ao = True,aron=True,rsi= True):
        
    for index in range(1,4):
        data = feat.add_ohlc_features(data, f'C{index}O{index}')
        data = feat.add_ohlc_features(data, f'C{index}O{index-1}')
        
        
    feat.add_ohlc_features(data, 'OH1')
    feat.add_ohlc_features(data, 'OL1')

    if rsi:
        data.ta.rsi(append=True)
        data[f'RSI_{14}']=data[f'RSI_{14}'].shift(1)

    if diff_ll:
        data = feat.add_hh_ll(data,periods=3,shift=1,drop_na=False)
        data = feat.add_hh_ll(data,periods=5,shift=1,drop_na=False)
        data = feat.add_hh_ll(data,periods=10,shift=1,drop_na=False)
        data = feat.add_hh_ll(data,periods=15,shift=1,drop_na=False)
        data['LL5LL3'] = 10000*(data['LL5'] - data['LL3'])
        data['LL10LL3'] = 10000*(data['LL10'] - data['LL3'])
        data['LL15LL3'] = 10000*(data['LL15'] - data['LL3'])
        data['HH5HH3'] = 10000*(data['HH5'] - data['HH3'])
        data['HH10HH3'] = 10000*(data['HH10'] - data['HH3'])
        data['HH15HH3'] = 10000*(data['HH15'] - data['HH3'])
        data["OLL3"] = 10000*(data["open"]-data["LL3"])
        data["OLL5"] = 10000*(data["open"]-data["LL5"])
        data["OLL10"] = 10000*(data["open"]-data["LL10"])
        data["OLL15"] = 10000*(data["open"]-data["LL15"])
        data["OHH3"] = 10000*(data["open"]-data["HH3"])
        data["OHH5"] = 10000*(data["open"]-data["HH5"])
        data["OHH10"] = 10000*(data["open"]-data["HH10"])
        data["OHH15"] = 10000*(data["open"]-data["HH15"])

    if use_bol:
        data['MA20'] = data['close'].shift(1).rolling(window=20).mean() 
        data['MA20dSTD'] = data['close'].shift(1).rolling(window=20).std() 
        data['Bollinger_Upper'] = data['MA20'] + (data['MA20dSTD'] * 2)
        data['Bollinger_Lower'] = data['MA20'] - (data['MA20dSTD'] * 2)


        data['U_minus_L'] = data['Bollinger_Upper'] - data['Bollinger_Lower']
        data['pullback'] = data['close'].shift(1) - data['Bollinger_Upper'].shift(1)
        data['throwback'] = data['close'].shift(1) - data['Bollinger_Lower'].shift(1)

    if rol_means:
        data['CO'] = data['close'] - data['open']
        data['rmCO(3)'] = data['CO'].shift(1).rolling(window=3).mean() 
        data['rmCO(4)'] = data['CO'].shift(1).rolling(window=4).mean() 
        data['rmCO(5)'] = data['CO'].shift(1).rolling(window=5).mean() 
        data['rmCO(6)'] = data['CO'].shift(1).rolling(window=6).mean() 

    if adx:
        for i in [5,7,10]:
            data.ta.adx(i,append=True)
            data[f'ADX_{i}'] = data[f'ADX_{i}'].shift(1)
            data[f'DMP_{i}'] = data[f'DMP_{i}'].shift(1)
            data[f'DMN_{i}'] = data[f'DMN_{i}'].shift(1)

    if ao :
        data.ta.ao(3,6,append=True)
        data[f'AO_{3}_{6}'] = data[f'AO_{3}_{6}'].shift(1)

    if kama:
        data.ta.kama(length=10,slow=30,append=True)
        data[f'OKAMA_{10}_{2}_{30}'] = data['open'] - data[f'KAMA_{10}_{2}_{30}'].shift(1)

    if macd :
        fast_macd=12
        slow_macd=26
        signal_macd =9
        data.ta.macd(fast=fast_macd,slow=slow_macd,signal=signal_macd,append=True)
        data[f'MACD_{fast_macd}_{slow_macd}_{signal_macd}'] = data[f'MACD_{fast_macd}_{slow_macd}_{signal_macd}'].shift(1)
        data[f'MACDh_{fast_macd}_{slow_macd}_{signal_macd}'] = data[f'MACDh_{fast_macd}_{slow_macd}_{signal_macd}'].shift(1)
        data[f'MACDs_{fast_macd}_{slow_macd}_{signal_macd}'] = data[f'MACDs_{fast_macd}_{slow_macd}_{signal_macd}'].shift(1)

    if aron:
        lenght_aroon = 7
        data.ta.aroon(length=lenght_aroon,append=True)
        data[f'AROONOSC_{lenght_aroon}'] = data[f'AROONOSC_{lenght_aroon}'].shift(1)
        data[f'AROOND_{lenght_aroon}'] = data[f'AROOND_{lenght_aroon}'].shift(1)
        data[f'AROONU_{lenght_aroon}'] = data[f'AROONU_{lenght_aroon}'].shift(1)


    data.dropna(inplace=True)
    features = []
    if rsi:
        features.append(f'RSI_{14}')
    if ao:
        features.append(f'AO_{3}_{6}')
    if kama:
        features.append(f'OKAMA_{10}_{2}_{30}')
    features.append("weekday")
    if macd :
        features.append(f'MACD_{fast_macd}_{slow_macd}_{signal_macd}')
        features.append(f'MACDh_{fast_macd}_{slow_macd}_{signal_macd}')
        features.append(f'MACDs_{fast_macd}_{slow_macd}_{signal_macd}')
    if aron:
        features.append(f'AROONOSC_{lenght_aroon}')
        features.append(f'AROOND_{lenght_aroon}')
        features.append(f'AROONU_{lenght_aroon}')
    features.append("C1O1")
    features.append("C2O2")

    if rol_means:
        features.append('rmCO(3)')
        features.append('rmCO(4)')
        features.append('rmCO(5)')
        features.append('rmCO(6)')
    features.append("C1O")
    features.append("C2O1")
    features.append("C3O3")
    features.append("C3O2")
    if diff_ll:
        features.append("LL5LL3")
        features.append("LL10LL3")
        features.append("LL15LL3")
        features.append("HH5HH3")
        features.append("HH10HH3")
        features.append("HH15HH3")
        features.append("OLL3")
        features.append("OLL5")
        features.append("OLL10")
        features.append("OLL15")
        features.append("OHH3")
        features.append("OHH5")
        features.append("OHH10")
        features.append("OHH15")
        
    features.append("OH1")
    features.append("OL1")
    if use_bol:
        features.append("U_minus_L")
        features.append("pullback")
        features.append("throwback")
    if adx:
        for i in [5]:
            features.append(f'ADX_{i}')
            #if i != 7:
            features.append(f'DMP_{i}')
            features.append(f'DMN_{i}')

    #features = list(data.columns[7:]) + ['weekday']

    return data, features
df, features = create_features(df, adx = True,use_bol = True,macd = True,diff_ll = True,rol_means = True,kama = True,ao = True,aron=True,rsi= True)

hull_period = 14  # Choisir une période appropriée
df.ta.hma(length=hull_period, append=True)
features.append(f'HMA_{hull_period}')
ema_period = 20  # Choisir une période appropriée
df.ta.ema(length=ema_period, append=True)
features.append(f'EMA_{ema_period}')

# Ajouter l'oscillateur stochastique
stoch_period = 14  # Choisir une période appropriée
df.ta.stoch(length=stoch_period, append=True)
features.append(f'Stoch_{stoch_period}')

lookback = 16  #lookback : Number of bars used as lookback
horizon = 8
features=['ADX_10', 'ADX_5', 'ADX_7', 'DMP_5', 'DMP_10', 'RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3','EMA_20', 'HMA_14', 'Bollinger_Lower', 'Bollinger_Upper' ]  #'high', 'low',
to_date = '2023-01-10'
backtest_perc = 0.1
time_from = 1200
time_to = 1600
training_perc = 0.7
seed= 1000
scale_fnc = 'minmax'

df = df.dropna()

training_set = ai.create_training_set(df[:to_date],                                      
                                      lookback=lookback,
                                      features=features,
                                      backtest_percent=backtest_perc,
                                      time_filter=True,
                                      time_from=time_from,
                                      time_to=time_to,
                                      excluded_weekdays=[],
                                      label='Label_CloseUp_{}'.format(horizon),
                                      label_type='categorical',
                                      training_percent= training_perc,
                                      seed=seed,
                                      training_shuffle=False,
                                      scale=True,
                                      scale_method=scale_fnc,  ## scale_fnc
                                      scale_per_slice=True,
                                      scale_feature_range=(0,1),  ##Previously (0,1)
                                      plot_train=False,
                                      flatten = False,
                                      scale_y=False,
                                      rawcategorical=False,
                                      y_squeeze=True,
                                      print_report=True,
                                      as_series=True
                                     )
## Map train,val,test,bck from the returned tuple
X_train, X_val, X_test, y_train, y_val, y_test, y_datetime,X_bck, y_bck, y_bck_datetime, train_scalers, val_scalers, test_scalers, bck_scalers = training_set