# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import datetime
import itertools as itr
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import sqlalchemy as sql
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


def printTime(tag):
    """
    ログ出力用の関数
    """
    print(tag + ':', datetime.datetime.now())


printTime('Inputファイルを表示')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# -

# ### データベースの接続設定

# データベースの接続設定
#   Connect Stringは下の形式で記述する(mysql+pymysqlはmysqlにPyMySQLライブラリを使ってアクセスするという意味)
#   mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
connect = 'mysql+pymysql://docker:docker@houseprices_mysql/kaggle'
engine = sql.create_engine(connect, echo=False)

# #### 学習データの読み込み

# +
train = pd.read_csv('/kaggle/input/train.csv')
test = pd.read_csv('/kaggle/input/test.csv')

# データベースに学習データ、テストデータを格納（Dataframeのindexはカラムに含めず、テーブル自体を上書き）
train.to_sql('train', engine, index=False, if_exists='replace')
test.to_sql('test', engine, index=False, if_exists='replace')

# -

#  #### 特徴量作成

# +
# 解析対象とする数値列を設定
targets_num = ['1stFlrSF',
               '2ndFlrSF',
               'BsmtFinSF1',
               'Fireplaces',
               'FullBath',
               'GarageArea',
               'GarageCars',
               'GarageYrBlt',
               'GrLivArea',
               'MasVnrArea',
               'OverallQual',
               'OverallCond',
               'TotalBsmtSF',
               'TotRmsAbvGrd',
               'YearBuilt',
               'YearRemodAdd',
               # ここからカテゴリ変数を数値変換した変数
               'LotShape',
               'BsmtCond',
               'BsmtQual',
               'FireplaceQu',
               'KitchenQual',
               'PoolQC',
               'GarageQual',
               'MasVnrType'
               ]

# 解析対象とするカテゴリ列を設定
targets_category = [
    'MSSubClass',
    'MSZoning',
    'Street',
    'Alley',
    # 'LotShape',
    'LandContour',
    'Utilities',
    'LotConfig',
    'LandSlope',
    'Neighborhood',
    'Condition1',
    'Condition2',
    'BldgType',
    'HouseStyle',
    'RoofStyle',
    'RoofMatl',
    'Exterior1st',
    'Exterior2nd',
    # 'MasVnrType',
    'ExterQual',
    'ExterCond',
    'Foundation',
    # 'BsmtQual',
    # 'BsmtCond',
    'BsmtExposure',
    'BsmtFinType1',
    'BsmtFinType2',
    'Heating',
    'HeatingQC',
    'CentralAir',
    'Electrical',
    # 'KitchenQual',
    'Functional',
    # 'FireplaceQu',
    'GarageType',
    'GarageFinish',
    # 'GarageQual',
    'GarageCond',
    'PavedDrive',
    # 'PoolQC',
    'Fence',
    'MiscFeature',
    'SaleType',
    'SaleCondition']

# 解析対象とする数値列とカテゴリ列の列名を連結
targets = targets_num.copy()
targets.extend(targets_category)

# Dataframeの列の型がobjectのままだとLabelEncoderでエラーするため、strに変換
# ※変換処理後にinfo()で型を確認しても、objectと表示されるが内部的にはstrに変換されている。
train[targets_category] = train[targets_category].astype(str)
test[targets_category] = test[targets_category].astype(str)


# +

def addTotalGrSF(df):

    dv_ary = []
    for index, row in df.iterrows():

        dv_ary.append(row['1stFlrSF'] + row['2ndFlrSF'])

    # Seriesに変換
    dv = pd.Series(dv_ary).astype(float)
    # DataFrameと結合
    df['TotalGrSF'] = dv


addTotalGrSF(train)
addTotalGrSF(test)

# 2020/03/28　追加


def addSoldYM(df):

    dv_ary = []
    for index, row in df.iterrows():

        dv_ary.append(str(row['YrSold']) + '{:0>2}'.format(row['MoSold']))

    # Seriesに変換
    dv = pd.Series(dv_ary).astype(int)
    # DataFrameと結合
    df['SoldYM'] = dv


addSoldYM(train)
addSoldYM(test)


def calcMean_LotArea(df, target, dv_count):

    dv_ary = []
    for index, row in df.iterrows():

        dv_ary.append(dv_count[row[target]])

    # Seriesに変換
    dv = pd.Series(dv_ary).astype(float)
    # DataFrameと結合
    df[target + '_LotArea_Mean'] = dv


def addMean_LotArea(df1, df2, target):

    dv_count = df1.append(df2, ignore_index=True)

    # target毎のNLotAreaの平均を算出
    dv_count = dv_count.groupby(target)['LotArea'].mean()

    calcMean_LotArea(df1, target, dv_count)
    calcMean_LotArea(df2, target, dv_count)


addMean_LotArea(train, test, 'Neighborhood')
addMean_LotArea(train, test, 'Condition1')
addMean_LotArea(train, test, 'HouseStyle')
addMean_LotArea(train, test, 'MSSubClass')


def calcStd_LotArea(df, target, dv_count):

    dv_ary = []
    for index, row in df.iterrows():

        dv_ary.append(dv_count[row[target]])

    # Seriesに変換
    dv = pd.Series(dv_ary).astype(float)
    # DataFrameと結合
    df[target + '_LotArea_Std'] = dv


def addStd_LotArea(df1, df2, target):

    dv_count = df1.append(df2, ignore_index=True)

    # target毎のNLotAreaの標準偏差を算出
    dv_count = dv_count.groupby(target)['LotArea'].std()

    calcStd_LotArea(df1, target, dv_count)
    calcStd_LotArea(df2, target, dv_count)


addStd_LotArea(train, test, 'Neighborhood')
addStd_LotArea(train, test, 'Condition1')
addStd_LotArea(train, test, 'HouseStyle')
addStd_LotArea(train, test, 'MSSubClass')


def calcVar_LotArea(df, target, dv_count):

    dv_ary = []
    for index, row in df.iterrows():

        dv_ary.append(dv_count[row[target]])

    # Seriesに変換
    dv = pd.Series(dv_ary).astype(float)
    # DataFrameと結合
    df[target + '_LotArea_Var'] = dv


def addVar_LotArea(df1, df2, target):

    dv_count = df1.append(df2, ignore_index=True)

    # target毎のNLotAreaの分散を算出
    dv_count = dv_count.groupby(target)['LotArea'].var()

    calcVar_LotArea(df1, target, dv_count)
    calcVar_LotArea(df2, target, dv_count)


addVar_LotArea(train, test, 'Neighborhood')
addVar_LotArea(train, test, 'Condition1')
addVar_LotArea(train, test, 'HouseStyle')
addVar_LotArea(train, test, 'MSSubClass')


# +

def setAstype(train, test, colName, setType):

    train[colName] = train[colName].astype(setType)
    test[colName] = test[colName].astype(setType)


dec = {
    'LotShape': {
        'Reg': 4,
        'IR1': 3,
        'IR2': 2,
        'IR3': 1
    },
    'BsmtCond': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'BsmtQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'FireplaceQu': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'KitchenQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'PoolQC': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'GarageQual': {
        'Ex': 5,
        'Gd': 4,
        'TA': 3,
        'Fa': 2,
        'Po': 1,
        'NA': -9999
    },
    'MasVnrType': {
        'BrkCmn': 4,
        'BrkFace': 3,
        'CBlock': 2,
        'Stone': 1,
        'None': -9999
    },
}

train.replace(dec, inplace=True)
test.replace(dec, inplace=True)

# 型を設定（後の計算でのエラー回避のため）
setAstype(train, test, 'LotShape', 'float')
setAstype(train, test, 'BsmtCond', 'float')
setAstype(train, test, 'BsmtQual', 'float')
setAstype(train, test, 'FireplaceQu', 'float')
setAstype(train, test, 'KitchenQual', 'float')
setAstype(train, test, 'PoolQC', 'float')
setAstype(train, test, 'GarageQual', 'float')
setAstype(train, test, 'MasVnrType', 'float')

# +
# 2020/05/30 全体の平均値からの差を求める Start


def addMeanDiff(df1, df2, target):

    dv_ary1 = []
    dv_ary2 = []
    
    mean = df1[target].mean()
    
    for index, row in df1.iterrows():

        dv_ary1.append(row[target] - mean)

    for index, row in df2.iterrows():
        
        dv_ary2.append(row[target] - mean)
        
    # Seriesに変換
    dv1 = pd.Series(dv_ary1).astype(float)
    dv2 = pd.Series(dv_ary2).astype(float)
    
    # DataFrameと結合
    df1[target + '_MeanDiff'] = dv1
    df2[target + '_MeanDiff'] = dv2


for target_num in targets_num:
    addMeanDiff(train, test, target_num)
    
# 2020/05/30 全体の平均値からの差を求める End

# +
printTime('組み合わせ特徴量の作成開始')


def calcFeatureValue(df, targets_pair):

    ary_multi = []
    ary_div1 = []
    ary_div2 = []

    target1 = targets_pair[0]
    target2 = targets_pair[1]

    for index, row in df.iterrows():

        if np.isnan(row[target1]) or np.isnan(row[target2]) \
                or row[target1] == 0 or row[target2] == 0:
            ary_multi.append(np.log(1))
        else:
            ary_multi.append(np.log(row[target1] * row[target2]))

        if np.isnan(row[target2]) or row[target2] == 0:
            ary_div1.append(0)
        else:
            ary_div1.append(row[target1] / row[target2])

        if np.isnan(row[target1]) or row[target1] == 0:
            ary_div2.append(0)
        else:
            ary_div2.append(row[target2] / row[target1])

    # Seriesに変換
    dv_multi = pd.Series(ary_multi).astype(float)
    dv_div1 = pd.Series(ary_div1).astype(float)
    dv_div2 = pd.Series(ary_div2).astype(float)

    # DataFrameと結合
    df[target1 + '_multi_' + target2] = dv_multi
    df[target1 + '_div_' + target2] = dv_div1
    df[target2 + '_div_' + target1] = dv_div2


# 2変数の組み合わせのlistを取得
targets_pairs = list(itr.combinations(targets_num, 2))

for targets_pair in targets_pairs:
    calcFeatureValue(train, targets_pair)
    calcFeatureValue(test, targets_pair)

printTime('組み合わせ特徴量の作成終了')

# +
# 2020/05/30 Target encodingをする Start

# ラベルエンコーディング用に学習データとテストデータをunion
lavel_x = train.append(test, ignore_index=True)

# カテゴリ変数をlabel encodingする
for c in targets_category:
    # 学習データとテストデータに基づいてラベルを設定
    le = LabelEncoder()
    le.fit(lavel_x[c])
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

# 2020/05/30 Target encodingをする End
# -

# ### 編集した学習データ、テストデータをデータベースに格納する

# データベースに学習データ、テストデータを格納（テーブル自体を上書き）
train.to_sql('train_featured', engine, index=False, if_exists='replace')
test.to_sql('test_featured', engine, index=False, if_exists='replace')

# ### データベースから編集済の学習データ、テストデータを取得し、特徴量と目的変数に分ける

# +
# データベースから学習データ、テストデータを取得
train_featured = pd.read_sql("SELECT * FROM train_featured", con=engine)
test_featured = pd.read_sql("SELECT * FROM test_featured", con=engine)

# 学習データを特徴量と目的変数に分ける
train_x = train_featured.drop(['Id', 'SalePrice'], axis=1)
train_y = train_featured['SalePrice']

# テストデータは特徴量のみなので、そのままでよい
test_x = test_featured.drop(['Id'], axis=1)
test_id = test_featured['Id']
# -

# #### モデル作成とバリデーション
# LightGBMを使用してモデルを作成します。

# +
printTime('モデルの作成開始')

va_pred_list = []
va_weight_list = []
pred_list = []

# 学習データを学習データとバリデーションデータに分ける
kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):
    tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

    # 2020/05/30 Target encodingをする Start

    # 2020/05/30 Target encodingをする End

    # 特徴量と目的変数をlightgbmのデータ構造に変換する
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(va_x, va_y)

    # ハイパーパラメータの設定
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'num_leaves': 40,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l2': 2,
    }

    # 作成する決定木の数を指定
    num_round = 100

    # 学習の実行
    # カテゴリ変数をパラメータで指定している
    # バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=num_round,
                      # categorical_feature=targets_category,
                      valid_names=['train', 'valid'],
                      valid_sets=[lgb_train, lgb_eval],
                      verbose_eval=0)

    # バリデーションデータでのスコアの確認
    va_pred = model.predict(va_x)
    va_pred_list.append(va_pred)

    # バリデーションデータでのスコア(真の値の対数と予測値の対数の二乗平均平方根誤差 (RMSE))を計算する
    rmse = np.sqrt(mean_squared_error(np.log(va_y), np.log(va_pred)))
    print(f'RMSE: {rmse:.4f}')

    # 加重平均を求めるための重さを設定(rmseは小さい方がいいため、逆数を設定)
    va_weight_list.append(np.reciprocal(rmse))

    # 結果の可視化
    sns.jointplot(
        va_y,
        va_pred,
        kind="reg",
        xlim=(0, 600000),
        ylim=(0, 600000)
    ).set_axis_labels("true", "pred")

    # 2020/05/30 予想値と真の値の差が大きいデータを分析する Start

    # 2020/05/30 予想値と真の値の差が大きいデータを分析する End

    # 予測
    pred_list.append(model.predict(test_x))


va_pred_list = np.array(va_pred_list)
pred_list = np.array(pred_list)

print(va_weight_list)

# 提出用ファイルの作成
submission = pd.DataFrame({'Id': test_id, 'SalePrice': np.average(pred_list, axis=0, weights=va_weight_list)})
submission.to_csv('/kaggle/output/submission_ensemble.csv', index=False)

printTime('モデルの作成終了')
# -

# #### データの分析にどの特徴量が重要だったのかをプロット

lgb.plot_importance(model, figsize=(10, 30), max_num_features=100)

# #### 分析に使用した決定木を可視化

lgb.create_tree_digraph(model)
