import talib

MACD = lambda *args: talib.MACD(args[0], fastperiod = args[1], 
                                slowperiod = args[2], 
                                signalperiod = args[3]) # 12,26,9 # got three output params

RSI = lambda *args: talib.RSI(args[0], timeperiod = args[1]) # 14
OBV = lambda *args: talib.OBV(args[0], args[1])

def preprocessing_foo(func, *args):
    try:
        return list(func(*args).fillna(0))[-1]
    except:
         return 0