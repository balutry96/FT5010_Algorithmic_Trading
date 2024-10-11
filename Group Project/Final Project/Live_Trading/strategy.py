import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
import pandas as pd
import time
import logging

from datetime import datetime, timedelta
from ta.momentum import RSIIndicator

class strategy:
    def __init__(self, api_key, account_id, currencies, units, stoploss):
        self.api_key = api_key
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=api_key)
        self.long_list = []
        self.short_list = []
        self.started = False
        self.stop = False
        self.currencies = currencies
        self.units = units
        self.stoploss = stoploss
        self.count = 30 # count of data to fetch
        self.trade_id = 1
        self.trades = []
        self.active_pairs = []
        self.all_currencies = []
        self.fetch_account_instruments()
        self.benchmark_count = 0
        self.account_balance = pd.DataFrame([], columns=['Account Balance','Beginning Balance','Benchmark','timestamp'])
        
        # Configure logging
        logging.basicConfig(filename='trades.log', level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    def fetch_account_instruments(self):
        r = accounts.AccountInstruments(accountID=self.account_id)
        self.client.request(r)
        instruments = r.response['instruments']
        self.all_currencies = sorted([inst['name'] for inst in instruments if 'USD' in inst['name']])

    def fetch_data(self, currency, count):
        params = {"count": count, "granularity": "S10"}
        r = instruments.InstrumentsCandles(instrument=currency, params=params)
        self.client.request(r)
        return pd.DataFrame(r.response['candles'])

    def calculate_moving_averages(self, currency_data):
        currency_data['close'] = currency_data['mid'].apply(lambda x: float(x['c']))
        LTMA_Pre = currency_data.close[(self.count-1-21):self.count-1].mean() # LTMA Window = 21
        STMA_Pre = currency_data.close[(self.count-1-5):self.count-1].mean() # SMTA Window = 5
        LTMA_Current = currency_data.close[(self.count-21):self.count].mean()
        STMA_Current = currency_data.close[(self.count-5):self.count].mean()
        return LTMA_Pre, STMA_Pre, LTMA_Current, STMA_Current

    def calculate_macd(self, currency_data):
        currency_data['close'] = currency_data['mid'].apply(lambda x: float(x['c']))
        Short_EMA = currency_data.close.ewm(span=12, adjust=False).mean() # default MACD period
        Long_EMA = currency_data.close.ewm(span=26, adjust=False).mean() # default MACD period
        MACD = Short_EMA - Long_EMA
        Signal = MACD.ewm(span=9, adjust=False).mean() # default MACD period
        MACD_Pre = MACD[self.count-1]
        MACD_Current = MACD[self.count-2]
        Signal_Pre = Signal[self.count-1]
        Signal_Current = Signal[self.count-2]
        return MACD_Pre, Signal_Pre, MACD_Current, Signal_Current

    def calculate_rsi(self, currency_data):
        currency_data['close'] = currency_data['mid'].apply(lambda x: float(x['c']))
        RSI = RSIIndicator(close=currency_data.close).rsi()
        return RSI[self.count-1]

    def place_order(self, currency, units):
        data = {
            "order": {
                "instrument": currency,
                "units": units,
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        r = orders.OrderCreate(accountID=self.account_id, data=data)
        self.client.request(r)
        self.trade_id += 1
        logging.info(f"Placed order for {currency} - Units: {units}")
        self.update_trades(currency, units, "OPEN")

    def close_trade(self, currency):
        try:
            r = positions.PositionDetails(accountID=self.account_id, instrument=currency)
            self.client.request(r)
            position = r.response['position']
            data = {"longUnits": "ALL"}
            r = positions.PositionClose(accountID=self.account_id, instrument=currency, data=data)
            self.client.request(r)
            logging.info(f"Closed trade for {currency}")
            self.update_trades(currency, -position['long']['units'], "CLOSE")
        except oandapyV20.exceptions.V20Error as e:
            print("Error during close_trade")
            if e.code == "CLOSEOUT_POSITION_DOESNT_EXIST":
                logging.warning(f"Position for {currency} does not exist. Skipping closeout.")
            elif e.code == "NO_SUCH_POSITION":
                logging.warning(f"No such position for {currency}. Skipping closeout.")
            else:
                raise e

    def run(self):
        self.benchmark_count = int(100000/self.fetch_data('EUR_USD', count=1)['mid'].apply(lambda x: float(x['c']))[0])
        while not self.stop:
            for currency in self.currencies:
                currency_data = self.fetch_data(currency, count=self.count)
                LTMA_Pre, STMA_Pre, LTMA_Current, STMA_Current = self.calculate_moving_averages(currency_data)
                MACD_Pre, Signal_Pre, MACD_Current, Signal_Current = self.calculate_macd(currency_data)
                RSI = self.calculate_rsi(currency_data)
                logging.info(f"{currency} - LTMA_Pre: {LTMA_Pre}, STMA_Pre: {STMA_Pre}, LTMA_Current: {LTMA_Current}, STMA_Current: {STMA_Current}")
                logging.info(f"{currency} - MACD_Pre: {MACD_Pre}, Signal_Pre: {Signal_Pre}, MACD_Current: {MACD_Current}, Signal_Current: {Signal_Current}")
                logging.info(f"{currency} - RSI: {RSI}")
                
                # First Trade - entry
                if currency not in self.long_list and currency not in self.short_list:
                    logging.info(f"Found currency: {currency}")
                    # Checking Bullish Crossover
                    if STMA_Pre < LTMA_Pre and STMA_Current > LTMA_Current:
                        if MACD_Pre < Signal_Current and MACD_Current > Signal_Current:
                            if RSI < 30:
                                self.place_order(currency, units=self.units)  # Adjust units as needed
                                self.long_list.append(currency)
                    # Checking Bearish Crossover
                    if STMA_Pre > LTMA_Pre and STMA_Current < LTMA_Current:
                        if MACD_Pre > Signal_Current and MACD_Current < Signal_Current:
                            if RSI > 70:
                                self.place_order(currency, units=-self.units)  # Adjust units as needed
                                self.short_list.append(currency)
                
                # Exit and re-entry
                if currency in self.long_list:
                    # Checking Bearish Crossover
                    if STMA_Pre > LTMA_Pre and STMA_Current < LTMA_Current:
                        if MACD_Pre > Signal_Current and MACD_Current < Signal_Current:
                            if RSI > 70:
                                self.close_trade(currency)
                                self.long_list.remove(currency)
                                self.place_order(currency, units=-self.units)  # Adjust units as needed
                                self.short_list.append(currency)
                
                if currency in self.short_list:
                    # Checking Bullish Crossover
                    if STMA_Pre < LTMA_Pre and STMA_Current > LTMA_Current:
                        if MACD_Pre < Signal_Current and MACD_Current > Signal_Current:
                            if RSI < 30:
                                self.close_trade(currency)
                                self.short_list.remove(currency)
                                self.place_order(currency, units=self.units)  # Adjust units as needed
                                self.long_list.append(currency)
            
            # Check account balance and stop trading if below threshold
            balance = self.get_account_balance()
            if self.started:
                benchmark = self.benchmark_count*self.fetch_data('EUR_USD', count=1)['mid'].apply(lambda x: float(x['c']))[0]
                # Update account balance table
                self.account_balance.loc[len(self.account_balance.index)] = {
                    'Account Balance': balance,
                    'Beginning Balance': 100000,
                    'Benchmark': benchmark,
                    'timestamp': pd.Timestamp.now()
                }
            logging.info(f"Account balance: {balance}")
            if balance < self.stoploss:
                self.stop = True
                self.close_all_trades()
            
            time.sleep(10)  # Reduced sleep duration for testing purposes

    def get_account_balance(self):
        r = accounts.AccountSummary(accountID=self.account_id)
        self.client.request(r)
        balance = float(r.response['account']['balance'])
        return balance

    def close_all_trades(self):
        for currency in self.long_list + self.short_list:
            self.close_trade(currency)
        self.long_list = []
        self.short_list = []

    def get_pnl(self):
        r = accounts.AccountSummary(accountID=self.account_id)
        self.client.request(r)
        balance = float(r.response['account']['balance'])
        nav = float(r.response['account']['NAV'])
        unrealized_pl = float(r.response['account']['unrealizedPL'])
        realized_pl = nav - balance
        return unrealized_pl, realized_pl

    def update_trades(self, currency, units, order_type):
        trade = {
            'currency': currency,
            'units': units,
            'order_type': order_type,
            'timestamp': pd.Timestamp.now()
        }
        self.trades.append(trade)

    def update_active_pairs(self):
        self.active_pairs = self.long_list + self.short_list

    def get_open_positions(self):
        r = positions.OpenPositions(accountID=self.account_id)
        self.client.request(r)
        return r.response['positions']

    def deploy(self, duration):
        start_time = time.time()
        end_time = start_time + duration * 3600  # Convert duration to seconds
        while time.time() < end_time and not self.stop:
            self.run()
            self.update_active_pairs()
        self.close_all_trades()
        unrealized_pl, realized_pl = self.get_pnl()
        logging.info(f"Trading session completed. Unrealized P&L: {unrealized_pl}, Realized P&L: {realized_pl}")
				
    def start(self, duration):
        self.started = True
        self.deploy(duration)

##if __name__ == '__main__':
##    api_key = "0914bb39c52a7126fc3df770f2109342-f29f1b82bf06adfe4e904e81ed1512a4" # Kay's
##    account_id = "101-003-28603731-001"
##    currencies = ["EUR_USD", "USD_JPY", "USD_CNY"]
##    strategy = Strategy(api_key, account_id, currencies, units=5000, stoploss=85000)
##    strategy.start(duration=24)
