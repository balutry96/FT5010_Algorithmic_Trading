import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import pytz
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, ParameterGrid
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")# , category=UserWarning, module='lightgbm')

def fetch_all_data(client, instrument, granularity):
    # Calculate start and end times dynamically
    utc_now = datetime.now().replace(tzinfo=pytz.utc)
    end = utc_now - timedelta(days=1) 
    start = end - timedelta(days=6)
    print(f"start ==  {start} >>>>>>> end == {end}")
    data = []
    while start < end:
        next_step = min(star
t + timedelta(hours=6), end)  # Adjust the interval as needed
        params = {
            "from": start.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',  # Format start time
            "to": next_step.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',  # Format end time
            "granularity": granularity
        }
        print(f"start ==  {start} +++++ end == {next_step}")
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        fetched_data = r.response.get('candles', [])
        data.extend(fetched_data)
        start = next_step

    return data

def transform_data(data):
    transformed_data = []
    for currency_data in data:
        for currency, candles in currency_data.items():
            for candle in candles:
                transformed_candle = {
                    'timestamp': candle['time'],
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': candle['volume'],
                    'currency': currency
                }
                transformed_data.append(transformed_candle)
    
    df = pd.DataFrame(transformed_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.to_csv("data.csv", index=None)
    
    df.set_index('timestamp', inplace=True)
    df['currency'] = df['currency'].astype('category')
    
    # Create a new DataFrame to store the lagged features
    lagged_df = df.groupby('currency')['close'].shift(1)
    lagged_df.name = 'close_lag1'
    
    # Merge the lagged features with the original DataFrame
    df = pd.concat([df, lagged_df], axis=1)
    df.dropna(subset=['close_lag1'], inplace=True)
    
    return df


def get_raw_data(client, currencies):
    raw_data = []
    for currency in currencies:
        tmp = dict()
        print(f"Collecting {currency} data from OANDA")
        data = fetch_all_data(client, currency, granularity = 'M1')
        tmp[currency] = data
        raw_data.append(tmp)
    
    data = transform_data(raw_data)

    return data

class MovingAverageCrossoverStrategy:
    def __init__(self, window_short=5, window_long=21, start_index_short_pre=24, start_index_long_pre=8,
                 start_index_short_curr=25, start_index_long_curr=9):
        # Initialize with default values; these can be adjusted later.
        self.initial_position = 0  # Default to no initial position
        self.window_short = window_short
        self.window_long = window_long
        self.start_index_short_pre = start_index_short_pre
        self.start_index_long_pre = start_index_long_pre
        self.start_index_short_curr = start_index_short_curr
        self.start_index_long_curr = start_index_long_curr
        self.long_list = []
        self.short_list = []

    def set_parameters(self, params):
        # Set parameters dynamically based on optimization or input.
        self.window_short = params.get('window_short', self.window_short)
        self.window_long = params.get('window_long', self.window_long)
        self.start_index_short_pre = params.get('start_index_short_pre', self.start_index_short_pre)
        self.start_index_long_pre = params.get('start_index_long_pre', self.start_index_long_pre)
        self.start_index_short_curr = params.get('start_index_short_curr', self.start_index_short_curr)
        self.start_index_long_curr = params.get('start_index_long_curr', self.start_index_long_curr)
    
    def calculate_moving_averages(self, currency_data):
        end_index_short_pre = self.start_index_short_pre + self.window_short
        end_index_long_pre = self.start_index_long_pre + self.window_long
        end_index_short_curr = self.start_index_short_curr + self.window_short
        end_index_long_curr = self.start_index_long_curr + self.window_long

        MA_Long_Pre = currency_data['close'][self.start_index_long_pre:end_index_long_pre].mean()
        MA_Short_Pre = currency_data['close'][self.start_index_short_pre:end_index_short_pre].mean()
        MA_Long_Current = currency_data['close'][self.start_index_long_curr:end_index_long_curr].mean()
        MA_Short_Current = currency_data['close'][self.start_index_short_curr:end_index_short_curr].mean()

        #print(f"MA_Long_Pre: {MA_Long_Pre}, MA_Short_Pre: {MA_Short_Pre}, MA_Long_Current: {MA_Long_Current}, MA_Short_Current: {MA_Short_Current}")
        return MA_Long_Pre, MA_Short_Pre, MA_Long_Current, MA_Short_Current

    def generate_signal(self, currency_data):
        # Generate trading signals based on moving averages.
        ma_values = self.calculate_moving_averages(currency_data)
        if not ma_values:
            return None  # Early exit if we can't calculate MAs

        MA_Long_Pre, MA_Short_Pre, MA_Long_Current, MA_Short_Current = ma_values
        currency = currency_data['currency'].iloc[-1]

        # Decision logic to enter or exit trades.
        if MA_Short_Pre < MA_Long_Pre and MA_Short_Current > MA_Long_Current:
            if currency not in self.long_list:
                self.long_list.append(currency)
                return 'buy'
        elif MA_Short_Pre > MA_Long_Pre and MA_Short_Current < MA_Long_Current:
            if currency not in self.short_list:
                self.short_list.append(currency)
                return 'sell'

        # Manage ongoing positions
        if currency in self.long_list and MA_Short_Current < MA_Long_Current:
            self.long_list.remove(currency)
            return 'sell'
        elif currency in self.short_list and MA_Short_Current > MA_Long_Current:
            self.short_list.remove(currency)
            return 'buy'

        return None

class EventBasedBacktester:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
        self.grid_search_details = [] 

    def split_data(self, test_size=0.2):
        results = {}
        tscv = TimeSeriesSplit(n_splits=int(1 / test_size))
        
        for currency in self.data['currency'].cat.categories:
            currency_data = self.data[self.data['currency'] == currency]
            print(f"Processing currency: {currency}")
            for train_index, test_index in tscv.split(currency_data):
                train_data = currency_data.iloc[train_index]
                test_data = currency_data.iloc[test_index]
                print(f"Train range for {currency}: {train_data.index.min()} to {train_data.index.max()}")
                print(f"Test range for {currency}: {test_data.index.min()} to {test_data.index.max()}")
                results[currency] = (train_data, test_data)
                break 
        return results
    
    def split_data(self, test_size=0.2):
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=int(1 / test_size))
        all_train_data = pd.DataFrame()
        all_test_data = pd.DataFrame()
        
        # Iterate over each currency
        for currency in self.data['currency'].cat.categories:
            currency_data = self.data[self.data['currency'] == currency]
            
            # Apply TimeSeriesSplit to each currency data
            for train_index, test_index in tscv.split(currency_data):
                train_data = currency_data.iloc[train_index]
                test_data = currency_data.iloc[test_index]
                
                print(f"Train range for {currency}: {train_data.index.min()} to {train_data.index.max()}")
                print(f"Test range for {currency}: {test_data.index.min()} to {test_data.index.max()}")
                # Collect all train and test sets across all currencies
                all_train_data = pd.concat([all_train_data, train_data])
                all_test_data = pd.concat([all_test_data, test_data])
                
                # Typically, only one set of splits is needed per currency in this setup
                break
        
        return all_train_data, all_test_data


    def optimize_parameters(self, train_data, param_grid):
        best_score = -np.inf
        best_params = {}
        ml_model = lgb.LGBMClassifier(verbosity=-1)

        # Ensure train_data is prepared correctly
        if 'close_lag1' not in train_data.columns:
            train_data['close_lag1'] = train_data['close'].shift(1)
            train_data.dropna(inplace=True)  # Ensure no NaN values

        strategy_params = {k: v for k, v in param_grid.items() if k.startswith('start_index') or k.startswith('window')}
        lgb_params = {k: v for k, v in param_grid.items() if k in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_alpha']}

        for strategy_param in ParameterGrid(strategy_params):
            self.strategy.set_parameters(strategy_param)
            simulated_returns = self.backtest(train_data)

            # Align features and labels
            labels = (simulated_returns >= 0).astype(int)
            # features = train_data.iloc[:len(simulated_returns)][['close', 'close_lag1']] 
            features = train_data.iloc[1:][['close', 'close_lag1']] 

            if len(features) != len(labels):
                print(f"Mismatch in data lengths: features {len(features)}, labels {len(labels)}")
                continue

            grid_search = GridSearchCV(ml_model, lgb_params, cv=TimeSeriesSplit(n_splits=3))
            grid_search.fit(features, labels)
            score = grid_search.best_score_

            self.grid_search_details.append({
                'params': {**strategy_param, **grid_search.best_params_},
                'score': score
            })

            print(f"Testing params: {strategy_param}, Score: {score}")
            if score > best_score:
                best_score = score
                best_params = strategy_param.copy()
                best_params.update(grid_search.best_params_)

        print(f"Best parameters: {best_params}, Best score: {best_score}")
        return best_params, best_score

    def backtest(self, data):
        """
        Perform event-based backtesting, generating trading signals and calculating returns.
        - Positions are initialized to zero, assuming no position is held at the start.
        - The strategy continues to hold the previous position if no new signal is generated.
        """
        positions = [0] * len(data)  # Initialize positions with zeros

        # Check if strategy desires a different initial position, e.g., from previous state
        # This is for illustration; actual implementation may vary based on strategy needs
        if hasattr(self.strategy, 'initial_position'):
            positions[0] = self.strategy.initial_position

        for i in range(1, len(data)):
            current_data = data.iloc[:i + 1]
            signal = self.strategy.generate_signal(current_data)
            if signal == 'buy':
                positions[i] = 1
            elif signal == 'sell':
                positions[i] = -1
            else:
                positions[i] = positions[i - 1]  # Hold previous position if no new signal

        data['positions'] = positions
        data['returns'] = data['close'].pct_change()
        strategy_returns = data['positions'].shift(1) * data['returns']

        # Handle NaN values at the start of the returns series
        returns = strategy_returns.dropna()
        print(f"Sample returns: {returns.head()}")  # Debug output to check returns

        return returns

    def calculate_performance_metrics(self, returns):
        """
        Calculate key performance metrics.
        """
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max()

        volatility = returns.std()

        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }
        return metrics
    
    def run(self, test_size=0.2, param_grid=None):
        """
        Execute the full backtesting process, including parameter optimization.
        """
        train_data, test_data = self.split_data(test_size)
        best_params, best_score = self.optimize_parameters(train_data, param_grid) if param_grid else (None, None)

        if best_params:
            self.strategy.set_parameters(best_params)

        train_returns = self.backtest(train_data)
        test_returns = self.backtest(test_data)

        train_metrics = self.calculate_performance_metrics(train_returns)
        test_metrics = self.calculate_performance_metrics(test_returns)

        return train_metrics, test_metrics, best_params, self.grid_search_details

# Example parameter grid combining both strategy and LightGBM parameters
param_grid = {
    #'n_estimators': [30, 70, 100],
    'learning_rate': [0.01, 0.1],
    #'reg_alpha': [0.01, 0.1],
    'max_depth': [10, 20],
    'num_leaves': [50, 70],
    'window_short': [3,4, 5, 6],
    'window_long': [7,8,9,10,11],
    'start_index_short_pre': [11, 17, 24],
    'start_index_long_pre': [5, 8, 13],
    'start_index_short_curr': [10, 16, 25],
    'start_index_long_curr': [4, 9, 12]
}

currencies = ["EUR_USD", "USD_JPY", "GBP_USD"]
api_key = "4e53e68ce363ad463cd0f2b8238e6139-10969d3810927f51d5d712de5e50f150"
account_id = "101-003-28593287-001"
client = oandapyV20.API(access_token=api_key)

# data = get_raw_data(currencies)

data = get_raw_data(client, currencies)
print(data.describe())
# print("All Data collected")
# data.to_csv("data.csv")
strategy = MovingAverageCrossoverStrategy()
backtester = EventBasedBacktester(data, strategy)
train_metrics, test_metrics, best_params, grid_search_details = backtester.run(test_size=0.2, param_grid=param_grid)

print("Training Metrics:", train_metrics)
print("Testing Metrics:", test_metrics)
print("Best Parameters:", best_params)