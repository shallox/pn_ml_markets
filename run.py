import os
from datetime import datetime, timedelta
from time import sleep
from aux_tools import config_parser, logging_config, check_os_type
import sys
import argparse
import subprocess


def out_packages():
    import requests
    import pandas as pd
    from sqlalchemy import create_engine, Column, Float, BigInteger, select, text, inspect
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sklearn.preprocessing import MinMaxScaler
    from keras.preprocessing.sequence import TimeseriesGenerator
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense
    from tqdm import tqdm

    eng = create_engine(
        config_parser(['settings', 'database', 'connection', 'data'], 'settings.yaml')
    )
    url = "https://api.binance.com/api/v3/klines"

    def add_data_to_db(data, symbol):
        """
        Commits data to databse.

        :param data:
        :param symbol: str
        :return:
        """
        engine = eng
        base = declarative_base()

        class Data(base):
            __tablename__ = symbol
            timestamp = Column(BigInteger, primary_key=True)
            open = Column(Float)
            high = Column(Float)
            low = Column(Float)
            close = Column(Float)
            volume = Column(Float)
            close_time = Column(BigInteger)
            quote_asset_volume = Column(Float)
            number_of_trades = Column(BigInteger)
            taker_buy_base_asset_volume = Column(Float)
            taker_buy_quote_asset_volume = Column(Float)

        base.metadata.create_all(engine)
        session = sessionmaker(bind=engine)()
        existing_record = engine.connect().execute(select(Data.timestamp)).all()
        ts_breakup = []
        for ts in existing_record:
            ts_breakup.append(ts[0])
        for candle_values in data.iterrows():
            candles = candle_values[1]
            if len(candles.keys()) == 11:
                if candles['Open time'] not in ts_breakup:
                    session.add(Data(
                        timestamp=candles['Open time'],
                        open=candles['Open'],
                        high=candles['High'],
                        low=candles['Low'],
                        close=candles['Close'],
                        volume=candles['Volume'],
                        close_time=candles['Close time'],
                        quote_asset_volume=candles['Quote asset volume'],
                        number_of_trades=candles['Number of trades'],
                        taker_buy_base_asset_volume=candles['Taker buy base asset volume'],
                        taker_buy_quote_asset_volume=candles['Taker buy quote asset volume'],
                    ))
        session.commit()

    def convert_to_dataframe(data):
        """
        Converts SL data to DataFrame.

        :param data:
        :return:
        """
        df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                         "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                         "Taker buy quote asset volume", "Ignore"])
        df.pop('Ignore')
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        if len(df) > 0:
            scaler = MinMaxScaler()
            try:
                df[["Open", "High", "Low", "Close", "Volume"]] = scaler.fit_transform(
                    df[["Open", "High", "Low", "Close", "Volume"]])
            except ValueError as ve:
                logging_config(ve, 1)
        return df

    def get_all_symbols_on_binance():
        """
        Returns a list of all market symbols available on binance.
        :return: list
        """
        symbresp = requests.get('https://api.binance.com/api/v3/exchangeInfo')
        symbols = [d['symbol'] for d in symbresp.json()['symbols']]
        return symbols

    def get_data(symbol_list, interval, hour_val):
        """
        Returns all data relating to markets.
        :return:
        """

        start_time = datetime.now()
        tik_gd = 0
        for market in tqdm(symbol_list, desc="Market data collection progress: "):
            tik_gd += 1
            params = {
                "interval": interval,
                "symbol": market,
                'startTime': int((datetime.now() - timedelta(hours=int(hour_val))).timestamp() * int(hour_val))
            }
            response = requests.get(url, params=params)
            data = response.json()
            data = convert_to_dataframe(data)
            add_data_to_db(data, market)
            fst = start_time + timedelta(minutes=1)
            if tik_gd >= 1199 and start_time < fst:
                sleep((start_time - fst).seconds)
                start_time = datetime.now()
                tik_gd = 0
            elif tik_gd >= 1199 or start_time >= fst:
                start_time = datetime.now()
                tik_gd = 0

    def train_on_dataset(target, name_m, epo, time_steps):
        """
        Trains on data set.
        :param target:
        :param name_m:
        :param epo:
        :param time_steps:
        :return:
        """
        df = eng.connect().execute(text(f'SELECT * FROM {target}'))
        df = pd.DataFrame(df.fetchall(), columns=list(df.keys()))
        generator = TimeseriesGenerator(df[["open", "high", "low", "close", "volume"]].values, df["close"].values,
                                        length=time_steps, batch_size=1)
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(time_steps, 5)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(generator, epochs=epo)
        model.save(f"models/{name_m}_{str(datetime.now()).replace(':', '-').replace(' ', '_').split('.')[0]}.h5")

    def predict_future(model_name_pred, target_dataset, time_step):
        # Load the saved model
        prediction_start_time = datetime.now()
        model = load_model(f'models/{model_name_pred}')

        # Load the market data
        print(target_dataset)

        params = {
            "interval": "1h",
            "symbol": target_dataset.replace(' ', ''),
            'startTime': int((datetime.now() - timedelta(hours=2)).timestamp() * int(hours_val))
        }
        response = requests.get(url, params=params)
        data = response.json()
        market_data = convert_to_dataframe(data)
        market_data.fillna(value=0)
        test_sample = market_data.tail(2)
        test_sample.drop(test_sample.index[-1])
        generator = test_sample[["Open", "High", "Low", "Close", "Volume"]].values
        last_sample = market_data.tail(1)['Close']
        generate = test_sample.reshape((test_sample.size, int(time_step), 5))
        predictions = model.predict(generate)
        predictions = MinMaxScaler().inverse_transform(predictions)
        predicted_close_price = predictions[0]
        percent_differance = ((predicted_close_price - last_sample) / last_sample) * 100
        prediction_run_time = datetime.now() - prediction_start_time
        print(f'Prediction done in {prediction_run_time.seconds} seconds:\n'
              f'Actual close price: {last_sample}\n'
              f'Predicted close price: {predicted_close_price}\n'
              f'Accuracy percentage: {percent_differance}\n'
              f'Amount out: {predicted_close_price - last_sample}')

    if cron_data is not None:
        collect_cron = cron_data.split(',')
        print(collect_cron)
    else:
        while True:
            options_avalable = ['collect', 'train', 'predict']
            while True:
                print(f'Crypto ML started:\n'
                      f'Input collect in order to gather data.\n'
                      f'Input train to start training on a market.\n'
                      f'Input predict to start predict market shifts.')
                option_one = input('Input command: ')
                if option_one not in options_avalable:
                    break
                if option_one == 'collect':
                    base_symbol_list = get_all_symbols_on_binance()
                    print(f'Select markets to collect by separating each with a ,\n '
                          f'Input all to scan all available markets.')
                    markets = input('Markets: ')
                    tik_interval = input('Interval (1m, 1h, 1d, 1m, 1y): ')
                    hours_val = input('Hours prior to now to start data cap: ')
                    if markets == 'all':
                        get_data(base_symbol_list, tik_interval, hours_val)
                    else:
                        get_data(markets.split(','), tik_interval, hours_val)
                    collection = input(f'Continue to collect output with a background task?: y/n')
                    if collection == 'y':
                        if check_os_type()[0] == 'Windows':
                            task_timer = input('Task frequancy (EG: HOURLY, DAILY): ')
                            create_scheduled_task(f"python {os.path.abspath(__file__)}"
                                                  f" --collect-cron {markets}, "
                                                  f"{tik_interval}, "
                                                  f"{hours_val}",
                                                  task_timer)
                        else:
                            cron_timer = input('Cron timer (eg: every hour 0 * * * *): ')
                            create_cron_job(f"python {os.path.abspath(__file__)}"
                                            f" --collect-cron {markets}, "
                                            f"{tik_interval}, "
                                            f"{hours_val}",
                                            cron_timer)

                if option_one == 'train':
                    inspector = inspect(eng)
                    schemas = inspector.get_table_names()
                    tik = 0
                    main_out = ''
                    last_entry = schemas[len(schemas) - 1]
                    print('Symbol List')
                    for symb in schemas:
                        if symb == last_entry:
                            main_out += symb
                        elif tik == 15:
                            tik = 0
                            main_out += f'\n{symb}, '
                        else:
                            main_out += f'{symb}, '
                        tik += 1
                    print(main_out)
                    target_symb = input('Enter symbol: ')
                    if target_symb in schemas:
                        model_name = input('Name of model? ')
                        epo = int(input('Number of epoch?\ndefault 10: '))
                        time_steps = int(input('Number of time steps?\ndefault 3: '))
                        if time_steps == '':
                            time_steps = 3
                        if epo == '':
                            epo = 10
                        train_on_dataset(target_symb, model_name, epo, time_steps)
                    else:
                        print(f"{target_symb} doesn't exist...")
                        break
                if option_one == 'predict':
                    models_selector = os.listdir('models/')
                    model_list = ''
                    end_step = len(models_selector) - 1
                    tik_ms = 0
                    cout_a = 0
                    for file in models_selector:
                        if '.h5' in file:
                            if tik_ms == end_step:
                                model_list += f'{file}'
                            elif cout_a > 12:
                                cout_a = 0
                                model_list += f'{file},\n'
                            else:
                                model_list += f'{file}, '
                            tik_ms += 1
                            cout_a += 1
                        print(f'Lets begin by selecting a pre trained model.\n '
                              f'{model_list}')
                        model_selection = input(f'Select model: ')
                        select_market = input(f'Select the market you would like to make a prediction on: ')
                        timed_steps = input(f'Enter the timed step for model, specific to the one used to generate: ')
                        predict_future(model_selection, select_market, timed_steps)


def handle_errors(error_msg):
    install_deps = input(f'Looks as though this is your 1st run and or {error_msg} is missing,\n'
                         f' would you like to run requirements.txt? y\\n?')
    if install_deps == 'y':
        os.system(f'pip install -r requirements.txt')
        out_packages()
    else:
        print(f'Ok, well gl out there :).')
        sys.exit(0)


def create_cron_job(command, schedule):
    # Run the crontab command to edit the cron jobs
    subprocess.run(["crontab", "-e"], stdin=subprocess.PIPE)

    # Append the new cron job to the cron file
    cron_job = f"{schedule} {command}"
    subprocess.run(["echo", cron_job], stdout=subprocess.PIPE)


def create_scheduled_task(command, schedule):
    task_name = f"ml_data_collect_{str(datetime.now()).replace(' ', '').replace('.', '')}"
    if schedule == 'HOURLY':
        start_time = str(datetime.now() + timedelta(hours=1)).split(' ')[1].split('.')[0]
    else:
        start_time = str(datetime.now() + timedelta(days=1)).split(' ')[1].split('.')[0]
    subprocess.run(["schtasks", "/create", "/tn", task_name, "/tr", command, "/sc", schedule, "/st", start_time],
                   shell=True, check=True)


if __name__ == '__main__':
    argument_pass = argparse.ArgumentParser()
    argument_pass.add_argument("--collect-cron", help="Collects data on a time window based on info passsed.")
    argument_list = argument_pass.parse_args()
    cron_data = argument_list.collect_cron
    try:
        out_packages()
    except ModuleNotFoundError as mnfe:
        handle_errors(mnfe)
    except ImportError as ie:
        handle_errors(ie)
