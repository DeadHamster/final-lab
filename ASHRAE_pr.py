import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('C:/Kag/ashrae-energy-prediction'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.model_selection \
import KFold
import lightgbm as lgb

from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objects as go
init_notebook_mode(connected=True)

######################################################################################################################
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}
metadata = pd.read_csv("C:/Kag/ashrae-energy-prediction/building_metadata.csv",dtype=metadata_dtype)
metadata.info(memory_usage='deep')
######################################################################################################################

weather_dtype = {"site_id":"uint8"}
weather_train = pd.read_csv("C:/Kag/ashrae-energy-prediction/weather_train.csv",parse_dates=['timestamp'],dtype=weather_dtype)
weather_test = pd.read_csv("C:/Kag/ashrae-energy-prediction/weather_test.csv",parse_dates=['timestamp'],dtype=weather_dtype)
print (weather_train.info(memory_usage='deep'))
print ("-------------------------------------")
print (weather_test.info(memory_usage='deep'))

######################################################################################################################

train_dtype = {'meter':"uint8",'building_id':'uint16','meter_reading':"float32"}
train = pd.read_csv("C:/Kag/ashrae-energy-prediction/train.csv",parse_dates=['timestamp'],dtype=train_dtype)
test_dtype = {'meter':"uint8",'building_id':'uint16'}
test_cols_to_read = ['building_id','meter','timestamp']
test = pd.read_csv("C:/Kag/ashrae-energy-prediction/test.csv",parse_dates=['timestamp'],usecols=test_cols_to_read,dtype=test_dtype)

######################################################################################################################

Subm = pd.DataFrame(test.index,columns=['row_id'])

########################################### MISING VALUE #############################################################
#MISING VALUE

missing_weather = pd.DataFrame(weather_train.isna().sum()/len(weather_train),columns=["Weather_Train_Missing_Pct"])
missing_weather["Weather Test Missing Value"] = weather_test.isna().sum()/len(weather_test)
missing_weather

######################################################################################################################
#узнаем, по каким параметрам у нас слишком много пропущенных данных
metadata.isna().sum()/len(metadata)

######################################################################################################################

metadata['floor_count_isNa'] = metadata['floor_count'].isna().astype('uint8')
metadata['year_built_isNa'] = metadata['year_built'].isna().astype('uint8')
metadata.drop('floor_count',axis=1,inplace=True)

######################################################################################################################

missing_train_test = pd.DataFrame(train.isna().sum()/len(train),columns=["Missing_Pct_Train"])
missing_train_test["Missing_Pct_Test"] = test.isna().sum()/len(test)
missing_train_test
#нет пропущенных в train/test

############################################## TRAIN DATA #############################################################

train.describe(include='all')

######################################################################################################################

train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

######################################################################################################################

trace1 = go.Bar(x=train['meter'].unique(),y=train['meter'].value_counts().values,marker=dict(color="rgb(55, 83, 109)"),text='train')
trace2 = go.Bar(x=test['meter'].unique(),y=test['meter'].value_counts().values,marker=dict(color="blue"),text='test')
data=[trace1,trace2]
layout = go.Layout(title='Countplot of meter',xaxis=dict(title='Meter'),yaxis=dict(title='Count'),hovermode='closest')
figure = go.Figure(data=data,layout=layout)
iplot(figure)

######################################################################################################################

train.groupby('meter')['meter_reading'].agg(['min','max','mean','median','count','std'])

######################################################################################################################
#для анализа было информативно сделать такие манипуляции

for df in [train, test]:
    df['Month'] = df['timestamp'].dt.month.astype("uint8")
    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")
    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

######################################################################################################################

train[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
train[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Graph of Average Meter Reading")

######################################################################################################################

meter_Electricity = train[train['meter'] == "Electricity"]
meter_Electricity[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_Electricity[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Average Meter Readingfor Electricity Meter")

######################################################################################################################

meter_ChilledWater = train[train['meter'] == "ChilledWater"]
meter_ChilledWater[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_ChilledWater[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Average Meter Readingfor ChilledWater Meter")

######################################################################################################################

meter_Steam = train[train['meter'] == "Steam"]
meter_Steam[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_Steam[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Average Meter Readingfor Steam Meter")

######################################################################################################################

meter_HotWater = train[train['meter'] == "HotWater"]
meter_HotWater[['timestamp','meter_reading']].set_index('timestamp').resample("H")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Hour')
meter_HotWater[['timestamp','meter_reading']].set_index('timestamp').resample("D")['meter_reading'].mean().plot(kind='line',figsize=(10,6),label='Avg_Meter_by_Day')
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Average Meter Reading")
plt.title("Average Meter Readingfor HotWater Meter")

######################################################################################################################
#вычислили аутлайнера - №1099
train.groupby('building_id')['meter_reading'].agg(['count','min','max','mean','median','std'])
######################################################################################################################

train[train['building_id'] == 1099]['meter_reading'].describe()

######################################################################################################################
#удостоверились, что распределение нормальное. (если получается другое распределение на графике, то нельзя быть уверенными,
# что данные согласованы (допустим, одинаковой размерности))

sns.distplot(np.log1p(train['meter_reading']),kde=False)
plt.title("Distribution of Log of Meter Reading Variable")

######################################################################################################################
sns.distplot(np.log1p(train[train['meter'] == "Electricity"]['meter_reading']),kde=False)
plt.title("Distribution of Meter Reading per MeterID code: Electricity")
######################################################################################################################
sns.distplot(np.log1p(train[train['meter'] == "ChilledWater"]['meter_reading']),kde=False)
plt.title("Distribution of Meter Reading per MeterID code: Chilledwater")
######################################################################################################################
sns.distplot(np.log1p(train[train['meter'] == "Steam"]['meter_reading']),kde=False)
plt.title("Distribution of Meter Reading per MeterID code: Steam")
######################################################################################################################
sns.distplot(np.log1p(train[train['meter'] == "HotWater"]['meter_reading']),kde=False)
plt.title("Distribution of Meter Reading per MeterID code: Hotwater")
######################################################################################################################

metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",
                                "Retail":"Other","s1vices":"Other","Technology/science":"Other","Food sales and s1vice":"Other",
                                "Utility":"Other","Religious worship":"Other"},inplace=True)

################################ WEATHER DATA #########################################################################

weather_train.info(memory_usage='deep')
weather_test.info(memory_usage='deep')

######################################################################################################################

weather_train.isna().sum()/len(weather_train)

######################################################################################################################

def fill_weather_dataset(weather_df):
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"

    #полезно было понимать инф по таким распределениям
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df["week"] = weather_df["datetime"].dt.week
    weather_df["month"] = weather_df["datetime"].dt.month

    #сбрасываем индекс
    weather_df = weather_df.set_index(['site_id', 'day', 'month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['air_temperature'].mean(),
                                          columns=["air_temperature"])
    weather_df.update(air_temperature_filler, overwrite=False)

    #1
    cloud_coverage_filler = weather_df.groupby(['site_id', 'day', 'month'])['cloud_coverage'].mean()
    #2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'), columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler, overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['dew_temperature'].mean(),
                                          columns=["dew_temperature"])
    weather_df.update(due_temperature_filler, overwrite=False)

    #1
    sea_level_filler = weather_df.groupby(['site_id', 'day', 'month'])['sea_level_pressure'].mean()
    #2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'), columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler, overwrite=False)

    wind_direction_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_direction'].mean(),
                                         columns=['wind_direction'])
    weather_df.update(wind_direction_filler, overwrite=False)

    wind_speed_filler = pd.DataFrame(weather_df.groupby(['site_id', 'day', 'month'])['wind_speed'].mean(),
                                     columns=['wind_speed'])
    weather_df.update(wind_speed_filler, overwrite=False)

    #1
    precip_depth_filler = weather_df.groupby(['site_id', 'day', 'month'])['precip_depth_1_hr'].mean()
    # 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'), columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler, overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime', 'day', 'week', 'month'], axis=1)
    return weather_df

######################################################################################################################
weather_train = fill_weather_dataset(weather_train)
weather_test = fill_weather_dataset(weather_test)
######################################################################################################################
cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']
for col in cols:
    print (" Minimum Value of {} column is {}".format(col,weather_train[col].min()))
    print (" Maximum Value of {} column is {}".format(col,weather_train[col].max()))
    print ("----------------------------------------------------------------------")
######################################################################################################################

for df in [weather_train,weather_test]:
    df['air_temperature'] = df['air_temperature'].astype('float32')
    df['cloud_coverage'] = df['cloud_coverage'].astype('float16')
    df['dew_temperature'] = df['dew_temperature'].astype('float16')
    df['precip_depth_1_hr'] = df['precip_depth_1_hr'].astype('float32')
    df['sea_level_pressure'] = df['sea_level_pressure'].astype('float32')
    df['wind_direction'] = df['wind_direction'].astype('float32')
    df['wind_speed'] = df['wind_speed'].astype('float16')

######################################### MERGING ###################################################################

train = pd.merge(train,metadata,on='building_id',how='left')
test  = pd.merge(test,metadata,on='building_id',how='left')
print ("Training Data Shape {}".format(train.shape))
print ("Testing Data Shape {}".format(test.shape))

######################################################################################################################

train = pd.merge(train,weather_train,on=['site_id','timestamp'],how='left')
test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')
print ("Training Data Shape {}".format(train.shape))
print ("Testing Data Shape {}".format(test.shape))

######################################################################################################################

train['meter_reading'] = np.log1p(train['meter_reading'])

######################################################################################################################
train.drop(['timestamp','year_built'],axis=1,inplace=True)
test.drop(['timestamp','year_built'],axis=1,inplace=True)
######################################################################################################################
#чекаем корреляцию
drlevel = 0.9

corr_matrix = train.corr().abs()
corr_matrix.head()
######################################################################################################################
more_corr = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
more_corr.head()
to_drop = [column for column in more_corr.columns if any(more_corr[column] > drlevel)]

######################################################################################################################
y = train['meter_reading']
train.drop('meter_reading',axis=1,inplace=True)

categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth','floor_count_isNa']

params = {'feature_fraction': 0.75,
          'bagging_fraction': 0.75,
          'objective': 'regression',
          'max_depth': 11,
          'learning_rate': 0.15,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'rmse',
          "verbosity": -1,
          'reg_alpha': 0.5,
          'reg_lambda': 0.5,
          'random_state': 47,
          "num_leaves": 31}
#поделили на 3 сета
kf = KFold(n_splits=3)
models = []
for train_index, test_index in kf.split(train):
    train_features = train.iloc[train_index]
    train_target = y.iloc[train_index]

    test_features = train.iloc[test_index]
    test_target = y.iloc[test_index]

    d_training = lgb.Dataset(train_features, label=train_target, categorical_feature=categorical_cols,
                             free_raw_data=False)
    d_test = lgb.Dataset(test_features, label=test_target, categorical_feature=categorical_cols, free_raw_data=False)

    model = lgb.train(params, train_set=d_training, num_boost_round=2000, valid_sets=[d_training, d_test],
                      verbose_eval=100, early_stopping_rounds=50)
    models.append(model)

######################################################################################################################
#от каких переменных самое большое влияние
s11 = pd.DataFrame(models[0].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
s11['Importance'].plot(kind='bar',figsize=(10,6))
######################################################################################################################
s12 = pd.DataFrame(models[1].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
s12['Importance'].plot(kind='bar',figsize=(10,6))
######################################################################################################################
s13 = pd.DataFrame(models[2].feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')
s13['Importance'].plot(kind='bar',figsize=(10,6))
######################################################################################################################
stepsize = 500000
results = np.zeros(test.shape[0])
for model in models:
  predictions = []
  for i in range(0, test.shape[0], stepsize):
    predictions.append(np.expm1(model.predict(test.loc[i:i+stepsize-1,:], num_iteration=model.best_iteration)))
  results += (1 / len(models)) * np.concatenate(predictions, axis=0)
  del(model)

######################################################################################################################

Subm['meter_reading'] = results
Subm['meter_reading'].clip(lower=0,more_corr=None,inplace=True)
Subm.to_csv("M.csv",index=None)

