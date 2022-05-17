import gensim
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
import tensorflow as tf
#import tensorflow.keras as keras
#from tensorflow.keras.layers import Dense,Dropout, Activation, Flatten
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.metrics import RootMeanSquaredError
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import Conv1D, MaxPooling1D
#from tensorflow.keras.layers import LSTM

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.layers import Conv1D, MaxPooling1D

#from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import VotingRegressor,GradientBoostingRegressor,StackingRegressor

from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.metrics import r2_score

import yfinance as yf
from yahoofinancials import YahooFinancials

from datetime import timedelta



with open (f"modeling/ticker_news_data.pickle","rb") as f:
    df_dict = pickle.load(f)
df_values = df_dict.values()
df = list(df_values)


lda_list=[]
for i in range(10):
  lda_list.append(df[i]["lemma"].to_list())
lda_list = [item for sublist in lda_list for item in sublist]

dictionary = gensim.corpora.Dictionary(lda_list)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(doc) for doc in lda_list]
id2word = dictionary.id2token

lda_model_normal_bow = gensim.models.ldamulticore.LdaMulticore(
    corpus=corpus[:-1],
    id2word=dictionary,
    num_topics=30,
    random_state=100,
    chunksize=1000,
    passes=10,
    workers = 8)

def get_df_topics(lda,corpus):
  topic = []
  all_topics = lda.get_document_topics(corpus)
  for doc_topics in all_topics:
    topic.append(doc_topics)
  return topic

def topics_to_df (df):
  df_end = pd.DataFrame()

  for i in range(30):
      df_end["top_"+str(i)]=[]

  frames = [df["topics"],df_end]
  df_end = pd.concat(frames).fillna(0)

  for i in range(len(df['topics'])):
      for j in range(len(df['topics'][i])):
          df_end["top_"+str(df['topics'][i][j][0])].loc[i] = df['topics'][i][j][1]
  return df.merge(df_end, left_index=True, right_index=True).drop(0,axis=1)


results = ['result_aapl', 'result_amzn', 'result_avgo',
           'result_fb', 'result_goog', 'result_ma',
           'result_msft', 'result_nvda', 'result_tsla',
           'result_v']
result_dict = {}
for result in results:
  with open(f'modeling/sentiment data/{result}.pickle', 'rb') as f:
      result_dict[result] = pickle.load(f)

for i in range(10):
  result_flat = [item for sublist in result_dict[results[i]] for item in sublist]
  df[i][["positive","negative","neutral"]] = result_flat

top_list = ['top_0', 'top_1', 'top_10', 'top_11', 'top_12', 'top_13', 'top_14',
       'top_15', 'top_16', 'top_17', 'top_18', 'top_19', 'top_2', 'top_20',
       'top_21', 'top_22', 'top_23', 'top_24', 'top_25', 'top_26', 'top_27',
       'top_28', 'top_29', 'top_3', 'top_4', 'top_5', 'top_6', 'top_7',
       'top_8', 'top_9']

for j in range(10):
    X = df[j][['top_0', 'top_1', 'top_10', 'top_11', 'top_12', 'top_13', 'top_14',
       'top_15', 'top_16', 'top_17', 'top_18', 'top_19', 'top_2', 'top_20',
       'top_21', 'top_22', 'top_23', 'top_24', 'top_25', 'top_26', 'top_27',
       'top_28', 'top_29', 'top_3', 'top_4', 'top_5', 'top_6', 'top_7',
       'top_8', 'top_9']]
    y = df[j][['positive','negative',"neutral"]]

    lm = LinearRegression(n_jobs=2).fit(X, y)
    positive_score = lm.coef_[0] - np.mean(lm.coef_[0])
    negative_score = lm.coef_[1] - np.mean(lm.coef_[1])
    neutral_score = lm.coef_[1] - np.mean(lm.coef_[1])

    l = (positive_score - negative_score) * (1 - neutral_score)
    k = (df[j]["positive"] - df[j]["negative"]) * (1 - df[j]['neutral'])

    for i in range(30):
        df[j][f"{top_list[i]}_ap1"] = df[j][top_list[i]] * (k[i] + l[i])
        df[j][f"{top_list[i]}_pos"] = df[j][top_list[i]] * (df[j]["positive"] + positive_score[i])
        df[j][f"{top_list[i]}_neg"] = df[j][top_list[i]] * (df[j]["positive"] + negative_score[i])
        df[j][f"{top_list[i]}_neu"] = df[j][top_list[i]] * (df[j]["positive"] + neutral_score[i])
        df[j][f"{top_list[i]}_average_adjusted"] = (df[j][f"{top_list[i]}_pos"] - df[j][f"{top_list[i]}_neg"]) * (
                    1 - df[j][f"{top_list[i]}_neu"])
        df[j][f"{top_list[i]}_average"] = (df[j][f"{top_list[i]}_pos"] - df[j][f"{top_list[i]}_neg"])


def subtract_days_from_date(date, days):
    subtracted_date = pd.to_datetime(date) - timedelta(days=days)
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")
    return subtracted_date


def add_days_to_date(date, days):
    added_date = pd.to_datetime(date) + timedelta(days=days)
    added_date = added_date.strftime("%Y-%m-%d")
    return added_date

ticker_list = ["AAPL","AMZN","AVGO","FB","GOOG","MA","MSFT","NVDA","TSLA","V"]
market_data = []
for i in range(10):
    market_data.append(yf.download(ticker_list[i],
                      start=subtract_days_from_date(df[i]['date'][len(df[i])-1].strftime("%Y-%m-%d")[:10],days=2),
                      end=add_days_to_date(date=df[i]['date'][0].strftime("%Y-%m-%d")[:10], days=10),
                      progress=True,))

def daily_return(df, market_data, days=0):
    daily_return = []
    market_days = market_data.index.strftime("%Y-%m-%d").to_list()
    for i in range(len(df)):
        if str(df['date'].iloc[i].to_pydatetime())[:10] in market_data.index:
            if str(df['date'].iloc[i].to_pydatetime())[11:19]<"14:30":
                daily_return.append(
                    (market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])+days]['Open'] -
                    market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])-1]['Close'])/
                    market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])-1]['Close'])
                #day_2_return.append(market_data.iloc[market_days.index(df.loc[i]['date_day']) + 1]['Close'] -
                #                    market_data.iloc[market_days.index(df.loc[i]['date_day'])]['Open'])
            elif str(df['date'].iloc[i].to_pydatetime())[11:19]>"21:00":
                daily_return.append(
                    (market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])+1+days]['Open'] -
                     market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])]['Close'])/
                    market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])]['Close'])
            else:
                daily_return.append(
                    (market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])+days]['Close'] -
                     market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])]['Open']) /
                    market_data.iloc[market_days.index(str(df['date'].iloc[i].to_pydatetime())[:10])]['Open'])

        else:
            date = [d for d in market_days if d> str(df['date'].iloc[i].to_pydatetime())[:10]][0]
            if str(df['date'].iloc[i].to_pydatetime())[11:19] < "14:30":
                daily_return.append(
                    (market_data.iloc[market_days.index(date)]['Open'] -
                     market_data.iloc[market_days.index(date)-1][
                         'Close']) /
                    market_data.iloc[market_days.index(date)-1]['Close'])
                # day_2_return.append(market_data.iloc[market_days.index(df.loc[i]['date_day']) + 1]['Close'] -
                #                    market_data.iloc[market_days.index(df.loc[i]['date_day'])]['Open'])
            elif str(df['date'].iloc[i].to_pydatetime())[11:19] > "21:00":
                daily_return.append(
                    (market_data.iloc[market_days.index(date)+1]['Open'] -
                     market_data.iloc[market_days.index(date)]['Close']) /
                    market_data.iloc[market_days.index(date)]['Close'])
            else:
                daily_return.append(
                    (market_data.iloc[market_days.index(date)]['Close'] -
                     market_data.iloc[market_days.index(date)]['Open']) /
                    market_data.iloc[market_days.index(date)]['Open'])

    return daily_return


def get_corrected_day_data(df):
    date_list = []
    df['date']=pd.to_datetime(df['date'])
    for i in range(len(df)):
        if str(df['date'].iloc[i].to_pydatetime())[11:19]>"14:30" and str(df['date'].iloc[i].to_pydatetime())[11:19]<"21":
            date_list.append(df['date'].iloc[i].strftime("%Y-%m-%d"))
        else:
            date_list.append(add_days_to_date(df['date'].iloc[i].strftime("%Y-%m-%d"),1))

    return date_list

df_w_market = []
for i in range(10):
    df[i]["daily_return"] = daily_return(df[i], market_data[i], days=0)
    df[i]["next_day_return"] = daily_return(df[i], market_data[i], days=1)
    df[i]["two_day_return"] = daily_return(df[i], market_data[i], days=2)
    #df[i]["five_day_return"] = daily_return(df[i], market_data[i], days=5)
    #df[i]["seven_day_return"] = daily_return(df[i], market_data[i], days=7)











#model = gensim.models.ldamodel.LdaModel.load("\modeling\lda_model_new_bow.model")
