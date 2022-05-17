import json
import requests
import pandas as pd
from pathlib import Path

class data():
    def __init__(self, api_key):
        self.api_key = api_key
        self.path = Path(__file__).parent.joinpath("trial_data")

    def save_news(self, ticker_list):
        for ticker in ticker_list:
            r = requests.get(
                f'https://eodhistoricaldata.com/api/news?api_token={self.api_key}&s={ticker}&offset=0&limit=1000')
            x = r.json()
            df = pd.DataFrame(x)
            last_date = df['date'][len(df) - 1][:10]
            for _ in range(0, 2):  # while last_date[:4] != "2015"
                r = requests.get(
                    f'https://eodhistoricaldata.com/api/news?api_token={self.api_key}&s={ticker}&offset=0&limit=1000&to={last_date}')
                x = r.json()
                df_temp = pd.DataFrame(x)
                df = pd.concat([df, df_temp], ignore_index=True)
                last_date = df['date'][len(df) - 1][:10]
            df.to_csv(f"{self.path}/{ticker}.csv")

    def process_news(self):
        # get different article datasets and process them into the correct format
        pass