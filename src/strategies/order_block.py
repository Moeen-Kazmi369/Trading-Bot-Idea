import pandas as pd
import numpy as np

class OrderBlockDetector:
    """
    Final Approved Order Block Strategy (Dynamic Root v1)
    Logic:
    - Pump Detection: Dynamic window (1-5 candles) with total body > 4.5x Batch Average.
    - OB Identification: The last Red candle at the base of the pump.
    - Sequential Low Rules: C1 Low < Red Low < C3 Low (For Bullish).
    """
    
    @staticmethod
    def get_candle_type(open_, high, low, close):
        body = abs(close - open_)
        range_ = high - low
        if range_ == 0: return "Doji"
        body_pct = body / range_
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if body_pct < 0.15: return "Doji"
        if body_pct < 0.4 and lower_wick > 1.3 * body and upper_wick < 0.3 * range_: return "Hammer"
        if body_pct < 0.45 and upper_wick > 0.3 * range_ and lower_wick > 0.3 * range_: return "Spinning Top"
        return "Normal"

    @classmethod
    def detect_all(cls, df):
        """
        Scans entire dataframe for Bullish and Bearish OB segments.
        Returns a dictionary of potential setup points.
        """
        df = df.copy()
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['body'] = abs(df['close'] - df['open'])
        df['avg_body'] = df['body'].rolling(window=50).mean()
        
        bullish_setups = []
        bearish_setups = []
        
        # 1-Hour Trend Bias (Internal Simulation)
        # Assuming 5m data - 12 samples = 1 hour
        df['ema50_1h'] = df['close'].ewm(span=600, adjust=False).mean() # 50 * 12 = 600 samples for roughly 1h bias
        
        for i in range(50, len(df) - 5):
            # 1H Trend Context (Optional: Can be strictly enforced if needed)
            is_bullish_context = df.iloc[i]['close'] > df.iloc[i]['ema50_1h']

            # BULLISH POI SEARCH (1-5 Candle windows)
            found_bull_momentum = False
            for win in range(1, 6):
                window = df.iloc[i:i+win]
                if all(window['close'] > window['open']):
                    if window['body'].sum() > 4.5 * df.iloc[i]['avg_body']:
                        found_bull_momentum = True
                        break
            
            if found_bull_momentum:
                # Find last Red at base
                for back in range(i, i - 5, -1):
                    if back < 1: break
                    c1, c2, c3 = df.iloc[back-1], df.iloc[back], df.iloc[back+1]
                    if c2['close'] < c2['open']: # RED
                        if c1['close'] > c1['open'] and c3['close'] > c3['open']: # GRG
                            if c2['low'] > c1['low'] and c3['low'] > c2['low']: # Staircase
                                kind = cls.get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                                bullish_setups.append({
                                    'index': back,
                                    'timestamp': c2['timestamp'],
                                    'type': 'BULLISH',
                                    'zone_h': c2['high'],
                                    'zone_l': c2['low'],
                                    'plus_point': kind
                                })
                                break

            # BEARISH POI SEARCH (1-5 Candle windows)
            found_bear_momentum = False
            for win in range(1, 6):
                window = df.iloc[i:i+win]
                if all(window['close'] < window['open']):
                    if window['body'].sum() > 4.5 * df.iloc[i]['avg_body']:
                        found_bear_momentum = True
                        break
            
            if found_bear_momentum:
                # Find last Green at base
                for back in range(i, i - 5, -1):
                    if back < 1: break
                    c1, c2, c3 = df.iloc[back-1], df.iloc[back], df.iloc[back+1]
                    if c2['close'] > c2['open']: # GREEN
                        if c1['close'] < c1['open'] and c3['close'] < c3['open']: # RGR
                            if c2['high'] < c1['high'] and c3['high'] < c2['high']: # Staircase
                                kind = cls.get_candle_type(c2['open'], c2['high'], c2['low'], c2['close'])
                                bearish_setups.append({
                                    'index': back,
                                    'timestamp': c2['timestamp'],
                                    'type': 'BEARISH',
                                    'zone_h': c2['high'],
                                    'zone_l': c2['low'],
                                    'plus_point': kind
                                })
                                break

        return {'bullish': bullish_setups, 'bearish': bearish_setups}
