import os
from datetime import datetime, timedelta
import sys
import argparse
import subprocess
import io
import zipfile
import concurrent.futures


def out_packages(real_fp):
    from aux_tools import config_parser, logging_config, check_os_type
    import requests
    import pandas as pd
    from sqlalchemy import create_engine, Column, Float, BigInteger, select, text, inspect, Text
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy.exc import OperationalError
    from sklearn.preprocessing import MinMaxScaler
    from keras.preprocessing.sequence import TimeseriesGenerator
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense
    from tqdm import tqdm

    eng = create_engine(
        config_parser(['settings', 'database', 'connection', 'data'], os.path.join(real_fp, 'settings.yaml')),
        pool_size=100,
        max_overflow=0,
        pool_recycle=True
    )
    url = "https://api.binance.com/api/v3/klines"

    def add_data_to_db(data, symbol, engine):
        """
        Commits data to databse.

        :param data:
        :param symbol: str
        :return:
        """

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
                if candles['open_time'] not in ts_breakup:
                    session.add(Data(
                        timestamp=candles['open_time'],
                        open=candles['open'],
                        high=candles['high'],
                        low=candles['low'],
                        close=candles['close'],
                        volume=candles['volume'],
                        close_time=candles['close_time'],
                        quote_asset_volume=candles['quote_asset_volume'],
                        number_of_trades=candles['number_of_trades'],
                        taker_buy_base_asset_volume=candles['taker_buy_base_asset_volume'],
                        taker_buy_quote_asset_volume=candles['taker_buy_quote_asset_volume'],
                    ))
        session.commit()

    def convert_to_dataframe(data):
        """
        Converts SL data to DataFrame.

        :param data:
        :return:
        """

        df = pd.read_csv(io.StringIO(data.decode('utf-8')),
                         names=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                "taker_buy_quote_asset_volume", "Ignore"])
        df.pop('Ignore')
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        if len(df) > 0:
            scaler = MinMaxScaler()
            try:
                df[["open", "high", "low", "close", "volume"]] = scaler.fit_transform(
                    df[["open", "high", "low", "close", "volume"]])
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

    def collect_market_data(market, tik_interval_gd, epoch_arg_split, epoch_arge_gd, engine):
        db_name = f'{market}_{tik_interval_gd}'
        try:
            latest_data = engine.connect().execute(
                text(f'SELECT open_time FROM {db_name} ORDER BY open_time DESC LIMIT 1')).fetchall()
            base_date_time_temp = datetime.fromtimestamp(int(latest_data[0][0])).strptime(latest_data[0][0], '%Y-%m')
        except OperationalError as oe:
            base_date_time_temp = '2017-10'
        base_year = int(epoch_arg_split[0])
        base_month = int(epoch_arg_split[1])
        base_date_time_temp = str(base_date_time_temp).split('-')
        base_date_time = datetime(year=base_year, month=base_month,
                                  day=1).strftime('%Y-%m')
        dts_split = base_date_time.split('-')
        dtn_split = datetime.now().strftime('%Y-%m').split('-')
        new_start_year = (int(dtn_split[0]) - int(dts_split[0])) + 1
        for a in range(new_start_year):
            base_url = f'https://data.binance.vision/data/spot/monthly/klines/{market}/' \
                       f'{tiker_interval}/{market}-{tiker_interval}-{base_date_time}.zip'
            market_data = requests.get(base_url)
            if '<Message>The specified key does not exist.</Message>' not in market_data.text:
                bdts = base_date_time.split("-")
                if int(bdts[0]) == 12:
                    base_date_time = f'{int(bdts[0]) + 1}-01'
                else:
                    base_date_time = f'{bdts[0]}-{int(bdts[1]) + 1}'
                with zipfile.ZipFile(io.BytesIO(market_data.content)) as zf:
                    csv_file = [f for f in zf.filelist if f.filename.endswith('.csv')][0]
                    csv_content = zf.read(csv_file)
                    data = convert_to_dataframe(csv_content)
                    add_data_to_db(data, db_name, engine)

    def get_data(epoch_arge_gd, market_target_list_gd, tik_interval_gd, db_workers):
        """
        Returns all data relating to markets.
        :return:
        """

        engine = create_engine(
            config_parser(['settings', 'database', 'connection', 'data'],
                          os.path.join(real_fp, 'settings.yaml')),
            pool_size=int(db_workers),
            max_overflow=0,
            pool_recycle=True
        )
        epoch_arg_split = epoch_arge_gd.split('-')
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(db_workers)) as executor:
            futures = [executor.submit(collect_market_data, market, tik_interval_gd, epoch_arg_split, epoch_arge_gd,
                                       engine)
                       for market in market_target_list_gd]
            for future in concurrent.futures.as_completed(futures):
                if future.done():
                    print(f"Market data collection finished: {future.result()}")

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
            "interval": "1m",
            "symbol": target_dataset.replace(' ', ''),
            'startTime': int((datetime.now() - timedelta(hours=2)).timestamp())
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
        collect_cron = cron_data.rsplit(',', 2)
        if collect_cron[0] == 'all':
            target_symbs = get_all_symbols_on_binance()
        else:
            target_symbs = collect_cron[0].replace(' ', '').split(',')
        open(f'{os.path.join(real_fp, "logs/last_cron.txt")}', 'a').write(
            f'@{str(datetime.now()).split(".")[0]} | job started for {collect_cron[0]}\n')
        get_data(target_symbs, collect_cron[1], collect_cron[2])
        open(f'{os.path.join(real_fp, "logs/last_cron.txt")}', 'a').write(
            f'@{str(datetime.now()).split(".")[0]} | job finished for {collect_cron[0]}\n')
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
                    epoch_arge = input(f'How many years do we go back, note 2017-10 is 1st data set?: ')
                    target_markets = input('Would you like to target specific markets,\n'
                                           'or collect all markets on binance?\n'
                                           'To back-fill all markets input all\n'
                                           'To back-fill selected markets, enter each one seperated\n'
                                           'by a comma , : ')
                    if target_markets == 'all':
                        market_target_list = base_symbol_list
                    else:
                        market_target_list = target_markets.replace(' ', '').split(',')
                    tiker_interval = input('What ticker interval are you looking for?\n'
                                           '1m, 1h, 1d, 1m, 1y : ')
                    workers = input('How many workers would you like to run: ')
                    get_data(epoch_arge, market_target_list, tiker_interval, workers)
                    collection = input(f'Continue to collect output with a background task?: y/n')
                    if collection == 'y':
                        if check_os_type()[0] == 'Windows':
                            task_timer = input('Task frequency (EG: HOURLY, DAILY): ')
                            create_scheduled_task(f"{sys.executable} {os.path.abspath(__file__)}"
                                                  f' --collect-cron "{market_target_list}, '
                                                  f'{tiker_interval}"',
                                                  task_timer)
                        else:
                            cron_timer = input('Cron timer (eg: every hour 0 * * * *): ')
                            create_cron_job(f"{sys.executable} {os.path.abspath(__file__)}"
                                            f' --collect-cron "{market_target_list}, '
                                            f'{tiker_interval}"',
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
        out_packages(real_fp)
    else:
        print(f'Ok, well gl out there :).')
        sys.exit(0)


def create_cron_job(command, schedule):
    subprocess.run(["crontab", "-e"], stdin=subprocess.PIPE)
    cron_job = f"{schedule} {command}"
    subprocess.run(["echo", cron_job], stdout=subprocess.PIPE)


def create_scheduled_task(command, schedule):
    task_name = f"ml_data_collect_{str(datetime.now()).replace(' ', '').replace('.', '')}"
    if schedule == 'HOURLY':
        start_time = str(datetime.now() + timedelta(hours=1)).split(' ')[1].split('.')[0]
    else:
        start_time = str(datetime.now() + timedelta(days=1)).split(' ')[1].split('.')[0]
    run_command = ["schtasks", "/create", "/tn", task_name.replace(':', '-'), "/tr", f"{command}", "/sc", schedule,
                   "/st", start_time]
    subprocess.run(run_command, shell=True, check=True)


if __name__ == '__main__':
    argument_pass = argparse.ArgumentParser()
    argument_pass.add_argument("--collect-cron", help="Collects data on a time window based on info passsed.")
    argument_list = argument_pass.parse_args()
    cron_data = argument_list.collect_cron
    real_fp = os.path.realpath(__file__).rsplit("\\", 1)[0].rsplit('/', 1)[0]
    open(f'{os.path.join(real_fp, "logs/last_cron.txt")}', 'a').write(
        f'@{str(datetime.now()).split(".")[0]} | Beginning of app. Cron data is {cron_data}\n')
    try:
        out_packages(real_fp)
    except ModuleNotFoundError as mnfe:
        handle_errors(mnfe)
    except ImportError as ie:
        handle_errors(ie)
