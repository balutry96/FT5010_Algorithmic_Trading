import dash
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
from strategy import strategy

def start_dashboard(self):
    app = dash.Dash(__name__)

    # Define the color scheme
    colors = {
        'background': '#f7f7f7',
        'title': '#38495b',
        'text': '#333333',
        'text-dark': '#f8f9fa',
        'primary': '#007bff',
        'secondary': '#6c757d',
        'success': '#28a745',
        'info': '#17a2b8',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }

    app.layout = html.Div([
        html.H1('Trading Dashboard', style={'color': colors['title']}),
        html.P('This is a live trading simulation through Oanda using a strategy consisting of MA Crossover, MACD, and RSI Divergence. '+\
               'MA Crossover uses two moving averages, a short-term and a long-term, to generate buy and sell signals based on their crossovers. '+\
               'MACD uses the difference of short-term and long-term exponential moving averages to generate buy and sell signals. '+\
               'RSI measures the speed and magnitude of recent price changes to determine if a security is overbought or oversold.', style={'font-size':'11px','color': colors['text']}),
        html.P('Users can choose which forex pairs to include in the demo.', style={'font-size':'11px','color': colors['text']}),
        html.P('The dashboard displays the account balance, trade history, and open positions.', style={'font-size':'11px','color': colors['text']}),
        dcc.Dropdown(
            id='currency-dropdown',
            options=[{'label': currency, 'value': currency} for currency in self.all_currencies],
            value=self.currencies,
            multi=True,
            style={'margin-top':'20px', 'margin-bottom':'20px'}
        ),
        html.Div(id='tabs-div', children=[
            dcc.Tabs([
                dcc.Tab(label='Account', children=[
                    html.Div(id='account-div', children=[
                        dcc.Graph(id='portfolio-value-graph'),
                        dcc.Graph(id='trades-graph')
                    ], style={'margin':'20px','overflow':'scroll'})
                ], className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='History', children=[
                    html.Div(id='trade-history', style={'margin':'20px','overflow':'scroll'})
                ], className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Positions', children=[
                    html.Div(id='open-positions', style={'margin':'20px','overflow':'scroll'})
                ], className='custom-tab', selected_className='custom-tab--selected'),
            ]),
        ], style={'margin-bottom':'20px'}),
        html.Div(id='account-balance', style={'color': colors['text'], 'margin':'20px'}),
        html.Button('Close All Positions', id='close-all-button', n_clicks=0, style={'margin':'20px','padding':'5px'}),
        dcc.Interval(
            id='interval-component',
            interval=1000,  # Update every 1 second
            n_intervals=0
        )
    ], style={'font-family': 'Arial, sans-serif', 'font-color': colors['text'], 'padding':'20px', 'background-color': colors['background']})

    @app.callback(Output('portfolio-value-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_value_graph(n):
        df = self.account_balance.copy()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            fig = {
                'data': [
                    {'x': df.index, 'y': df['Account Balance'], 'type': 'lines', 'name': 'Account Balance',
                     'line': {'shape':'linear', 'color':'#1788bb', 'width':2, 'dash':'solid'}}, # blue
                    {'x': df.index, 'y': df['Beginning Balance'], 'type': 'lines', 'name': 'Beginning Balance',
                     'line': {'shape':'linear', 'color':'#eb7266', 'width':2, 'dash':'dash'}}, # red
                    {'x': df.index, 'y': df['Benchmark'], 'type': 'lines', 'name':'Benchmark (EUR_USD)',
                     'line': {'shape':'linear', 'color':'#17bb9c', 'width':2, 'dash':'dash'}}, #green
                ],
                'layout': {
                    'title': 'Account Balance vs. Benchmark (EUR_USD)',
                    'xaxis': {'title': 'Timestamp'},
                    'yaxis': {'title': 'Balance (SGD)'}
                }
            }
            return fig
        else:
            return {}

    @app.callback(Output('trades-graph', 'figure'),
                  [Input('interval-component', 'n_intervals')])
    def update_trades_graph(n):
        df = pd.DataFrame(self.trades)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            fig = {
                'data': [{
                    'x': df.index,
                    'y': df['units'],
                    'type': 'scatter',
                    'mode': 'markers',
                    'marker': {'size': 10, 'color': ['#17bb9c' if units > 0 else '#eb7266' for units in df['units']]},
                    'text': df['currency'] + ' - ' + df['order_type']
                }],
                'layout': {
                    'title': 'Trades',
                    'xaxis': {'title': 'Timestamp'},
                    'yaxis': {'title': 'Units'}
                }
            }
            return fig
        else:
            return {}

    @app.callback(Output('trade-history', 'children'),
                  [Input('interval-component', 'n_intervals')])
    def update_trade_history(n):
        df = pd.DataFrame(self.trades)
        if not df.empty:
            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        else:
            return "No trade history available."

    @app.callback(Output('open-positions', 'children'),
                  [Input('interval-component', 'n_intervals')])
    def update_open_positions(n):
        positions = self.get_open_positions()
        if positions:
            data = []
            for position in positions:
                position_data = {
                    'instrument': position['instrument'],
                    'long_units': position.get('long', {}).get('units', ''),
                    'long_avg_price': position.get('long', {}).get('averagePrice', ''),
                    'short_units': position.get('short', {}).get('units', ''),
                    'short_avg_price': position.get('short', {}).get('averagePrice', '')
                }
                data.append(position_data)
            df = pd.DataFrame(data)
            return dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        else:
            return "No open positions."

    @app.callback(Output('account-balance', 'children'),
                  [Input('interval-component', 'n_intervals')])
    def update_account_balance(n):
        balance = self.get_account_balance()
        unrealized_pl, realized_pl = self.get_pnl()
        balance_text = f"Account Balance: {balance:,}\n\nUnrealized P/L: {unrealized_pl:,}\n\nRealized P/L: {realized_pl:,}"
        return dcc.Markdown(balance_text)

    @app.callback(Output('close-all-button', 'n_clicks'),
                  [Input('close-all-button', 'n_clicks')])
    def close_all_positions(n_clicks):
        if n_clicks > 0:
            self.close_all_trades()
        return n_clicks

    @app.callback(Output('currency-dropdown', 'value'),
                  [Input('currency-dropdown', 'value')])
    def update_currencies(selected_currencies):
        self.currencies = selected_currencies
        if not self.started:
            self.start(duration=24)
        return selected_currencies

    app.run_server(debug=True)

if __name__ == '__main__':
    api_key = "0914bb39c52a7126fc3df770f2109342-f29f1b82bf06adfe4e904e81ed1512a4" # Kay's
    account_id = "101-003-28603731-001"
    strategy = strategy(api_key, account_id, [], units=5000, stoploss=85000)
    start_dashboard(strategy)
