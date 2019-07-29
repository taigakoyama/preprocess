import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from statistics import mean, median, variance, stdev


# from yellowbrick.target import # 2つの値から角度を求めて、配列を返す関数
# def make_radian_row(pca_result):
#     rad = []
#     for r in pca_result:
#         rad.append(math.atan(r[0]/r[1]))
#
#     return rad
# from yellowbrick.features import _rank1_d, _rank2_d


def show_value_counts(df, adviser='off'):
    """
    データフレーム内の列に対し、category、string型(number型以外)が含まれる場合、その値と数を表示する。
    adviser引数に'on'を入力すると、カーディナリティーが20以上のものを提示する。
    カーディナリティーを調整することを推奨する。

    :param df: データフレーム
    :param adviser: アドバイザー機能の有無。デフォルトは'off'
    :return: なし
    """

    no_number_df = df.select_dtypes(exclude='number')
    rows = len(no_number_df.index)

    gather_list = []

    for column_name in no_number_df.columns:

        print_series = no_number_df[column_name].value_counts()
        series_count = len(print_series)

        if rows == series_count:
            print('%s : all unique' % column_name)
            print('--------------------------------------------------------------')
            continue

        print('(column _name)%s : (cardinality)%d : (Total)%s \n' % (
            column_name, len(print_series), sum(print_series.values)))
        print(print_series)
        print('--------------------------------------------------------------')

        if 20 < series_count:
            gather_list.append(column_name)

    if 'on' == adviser:
        print('should gather : %s' % gather_list)


def show_null_count(df, adviser='off', border=0.7):
    """
    データフレーム内の列に対し、nullが含まれる列を表示する。
    adviser引数に'on'を入力すると、欠損の割合がborderを超える列名を表示する。
    欠損が多すぎる列は削除することを推奨する。

    :param df: データフレーム
    :param adviser: アドバイザー機能の有無。デフォルトは'off'
    :param border: アドバイザー機能が有効な場合に有効。指定した欠損の割合を超えた列名を表示する。デフォルトは0.7
    :return:なし
    """

    null_info_series = pd.Series(df.isnull().sum(), name='null_count')
    types_series = pd.Series(df.dtypes, name='type')

    print_data_frame = pd.concat([null_info_series, types_series], axis=1)
    print_data_frame = print_data_frame[print_data_frame['null_count'] != 0]

    print(print_data_frame)
    #     display(print_data_frame)

    if 'on' == adviser:
        border_count = len(print_data_frame.index) * border
        advise_data_frame = print_data_frame[print_data_frame['null_count'] > border_count]

        print('--------------------------------------------------------------')
        print('should omit : %s' % advise_data_frame.index)


def get_prefecture_set_from_column(df, column_name):
    """
    データフレームの指定カラム内に含まれる都道府県市町村を分けてsetとして返却。
    誤入力には対応していないため、返却された値を必ず確認すること。

    https://qiita.com/zakuroishikuro/items/066421bce820e3c73ce9

    :param df: データフレーム
    :param column_name: 住所情報が格納された列名
    :return: 都道府県市町村set
    """

    pat = '(...??[都道府県])((?:旭川|伊達|石狩|盛岡|奥州|田村|南相馬|那須塩原|東村山|武蔵村山|羽村|十日町|上越|富山|野々市|' \
          '大町|蒲郡|四日市|姫路|大和郡山|廿日市|下>松|岩国|田川|大村|宮古|富良野|別府|佐伯|黒部|小諸|塩尻|玉野|周南)市|' \
          '(?:余市|高市|[^市]{2,3}?)郡(?:玉村|大町|.{1,5}?)[町村]|(?:余市|高市|[^市]{2,3}?)郡|(?:.{1,4}市)?[^町]{1,4}?区|' \
          '.{1,7}?[市町村])(.*)'
    #     pat = '(.+?[都道府県])(.+?[市区町村])(.+)'

    return_set = set()

    for row in df[column_name]:

        for text in re.split(pat, row):
            return_set.add(text)

    return_set.remove('')

    return return_set


def get_value_set_from_columns(df, column_name_list):
    """
    複数列に同様の内容が入力されている場合、一意な値のセットを返却する。

    ユースケース：
    １．一意な値が何であるかを取得する
    ２．get_dummiesではなく自前で 0 or 1のフラグを立てる。

    :param df: データフレーム
    :param column_name_list: 対象列のリスト
    :return: 一意なset
    """

    return_set = set()

    for col_name in column_name_list:
        return_set = return_set.union(df[col_name])

    return_set.remove(np.nan)

    return return_set


def get_value_set_by_cut(df, target_column):
    """
    単一列内に値が複数入っている場合に一意な値のsetを作成する。
    delimiterは","、 " "、 "/"。

    :param df: データフレーム
    :param target_column: 対象列
    :return: 一意なset
    """

    if not type(target_column) is str:
        print('please push me, single column name!')

    return_set = set()
    if type(df[target_column][0]) in (str, int, float, 'category'):
        origin_set = set(df[target_column].value_counts().index)
        split_word_list = {',', ' ', '/'}

        for word in origin_set:
            for split_word in split_word_list:
                for w in word.split(split_word):
                    return_set.add(w)
        return return_set.difference(origin_set)

    df[target_column].apply(lambda x: return_set.add(i) for i in x)

    return return_set


def add_ohe_from_set(_df, target_columns, value_set):
    """
    get_dummiesではなく、一意なsetを元に、対象の複数列の値を元に自前で 0 or 1のフラグを立てる。
    target_columnsの列名のリスト内に含まれる値を対象に、set数分、列を増加させる。

    :param _df: データフレーム
    :param target_columns: 検索対象列のリスト
    :param value_set: 値のset
    :return: データフレーム
    """

    df = _df.copy()
    df['add'] = ''

    for target in target_columns:
        df['add'] = df['add'] + df[target].fillna('_na_n')

    for val in value_set:
        new_column_name = target_columns[0] + '_' + val
        df[new_column_name] = df['add'].apply(lambda x: 1 if val in x else 0)
        df[new_column_name] = df[new_column_name].astype('int')

    del df['add']

    return df


def show_useless_category(df, threshold=0.05):
    """
    カテゴリ列に対して、特定の割合以下のカテゴリ値を表示する。
    そのカテゴリ類はまとめるなどするか、行を削除することの検討を推奨する。

    :param df: データフレーム
    :param threshold: 閾値(列内のカテゴリ値の数÷行数)
    :return: なし
    """

    non_number_data_frame = df.select_dtypes(exclude='number')
    rows = len(non_number_data_frame.index)
    df_threshold = rows * threshold

    for column_name in non_number_data_frame.columns:

        print_series = non_number_data_frame[column_name].value_counts()
        series_count = len(print_series)

        # binaryの場合を除くため、カテゴリが3以上の時のみ処理を行う
        if series_count >= 3:
            useless_category_list = []

            for i in range(series_count):

                if print_series[i] < df_threshold:
                    useless_category_list.append(print_series.index[i])

            if not not useless_category_list:
                print('column_name : %s\nuseless categories : %s' % (column_name, useless_category_list))
                print('--------------------------------------------------------------')


def remove_one_column_after_get_dummies(_df, dummy_columns_list):
    """
    get_dummiesでremove_firstしていないケースかつ、nanを有効にした場合、列をどれかひとつ削除する。
    _nanを含んだ場合、

    :param _df: データフレーム
    :param dummy_columns_list: ダミー変数化した列のリスト
    :return: dummy_columns_list分の列を削除したデータフレーム
    """
    df = _df.copy()

    for column_name in dummy_columns_list:
        column_name_list = list(df.loc[:, df.columns.str.contains(column_name)].columns)

        print('--------------------------------------------------------------')
        print('<----origin---->')
        print(column_name_list)

        print('➡️delete column')

        if (column_name + '_nan') in column_name_list:
            del df[column_name + '_nan']
            print(column_name + '_nan')
            continue

        elif (column_name + '_OTHERS') in column_name_list:
            del df[column_name + '_OTHERS']
            print(column_name + '_OTHERS')
            continue

        del_column_name = column_name_list[0]
        del df[del_column_name]

        print(del_column_name)

    return df


def add_static_column(_df, target_columns):
    """
    行単位で、target_columnsの列のリストに指定された値たちの統計情報を追加する。

    :param _df: データフレーム
    :param target_columns: 集計対象の列のリスト
    :return: 列追加後のデータフレーム
    """

    df = _df.copy()

    df['add'] = df[target_columns].apply(lambda x: [y for y in x.values if str(y) != 'nan' and str(y) != '0.0'], axis=1)

    df[target_columns[0] + '_mean'] = df['add'].apply(lambda x: mean(x))
    df[target_columns[0] + '_max'] = df['add'].apply(lambda x: max(x))
    df[target_columns[0] + '_min'] = df['add'].apply(lambda x: min(x))
    df[target_columns[0] + '_median'] = df['add'].apply(lambda x: median(x))
    df[target_columns[0] + '_count'] = df['add'].apply(lambda x: len(x))

    del df['add']

    return df


def show_hist(df, target_columns_list):
    """
    ざっくりとしたヒストグラムの可視化を行う

    :param df: データセット
    :param target_columns_list: 可視化対象列のリスト
    :return: なし
    """
    fig, axes = plt.subplots(nrows=len(target_columns_list), ncols=1, figsize=(10, 500), squeeze=True)

    for i, target_columns in enumerate(target_columns_list):
        axes[i].hist(df[target_columns])
        axes[i].set_title(target_columns)


def get_agg_val_dict(_df, target_columns, calc_columns, search='off'):
    """
    ターゲットに指定された列(oheの列)に対して、値が1である行を集約して保持するためのdictionaryを返す。
    辞書のKey : ターゲットの列
    辞書のValue : 下記データセット
    　データセットのindex : 算出対象列の列名
    　データセットのcolumn : ターゲットの列の値が1のindex

    ターゲットの列名のリストが多すぎる場合や、get_dummiesしている場合にsearchを'on'を指定すると、
    target_columns[0]の値を含んだ列をtarget_columnsとして扱う。

    :param _df: データフレーム
    :param target_columns: ターゲットの列名のリスト(oheの列)
    :param calc_columns: 算出対象の列名のリスト
    :param search: target_columns[0]を元に列名のサーチをするか否か。デフォルトは'off'
    :return: 各種統計情報算出を行うための辞書
    """
    df = _df.copy()
    return_dict = {}

    if 'on' == search and len(target_columns) == 1:
        target_columns = [column_name for column_name in list(df.columns) if target_columns[0] in column_name]
        target_columns.remove(target_columns[0])

    for target in target_columns:
        calc_dict_series = df[df[target] == 1][calc_columns].T.apply(lambda x: list(x))
        return_dict[target] = calc_dict_series

    all0_calc_dict_series = df[df[target_columns].apply(lambda x: sum(x) == 0, axis=1)][calc_columns].T.apply(
        lambda x: list(x))

    return_dict['none'] = all0_calc_dict_series

    return return_dict


def add_calc_columns(_df, target, aggr_value_dict):
    """
    get_agg_val_dict()と対で使用予定。
    get_agg_val_dictで得た情報を元にoheの統計情報をデータフレームに追加する。

    :param _df: データフレーム
    :param target: aggr_value_dictを作った際に指定したカラム名。最終的なカラム名の一部に利用される。
    :param aggr_value_dict: get_agg_val_dict()の戻り値
    :return: oheの統計情報追加後のデータフレーム
    """

    df = _df.copy()

    # target_columns = [columnName for columnName in list(df.columns) if target in columnName]
    add_columns = ['max', 'min', 'mean', 'median']  # 集計項目の追加時は、このリストとseriesを作成しているlambdaに対しても集計処理を追加する。
    dict_keys = list(aggr_value_dict.keys())

    # 各行毎に集計データを追加
    for index, row in df.iterrows():

        column_names = row[dict_keys][row[dict_keys] == 1].index

        # 算出用のリストを生成する
        calc_df = pd.DataFrame(index=dict_keys, columns=[])
        for column in column_names:

            if calc_df.empty:
                calc_df = aggr_value_dict[column]

            else:
                calc_df = pd.concat([calc_df, aggr_value_dict[column]], axis=1, join='inner')

        if calc_df.empty:
            calc_df = aggr_value_dict['none']

        # 算出用リストを集計したSeries
        add_df = calc_df.apply(lambda x: pd.Series([max(x), min(x), mean(x), median(x)], index=add_columns), axis=1)

        # 算出リストを行に追加していく
        for i, v in add_df.stack().iteritems():

            if 0 == index:
                df[target + '_' + i[0] + '_' + i[1]] = 0

            df.at[index, target + '_' + i[0] + '_' + i[1]] = v

    return df


def remove_ohe_auto(_df, threshold=0.03):
    """
    データフレームのohe列のうち、列がそのデータの特徴を表すのに適していないと判断した場合に削除する。
    0 or 1のうち、カーディナリティーが低い方の数がthresholdの割合を下回る時に削除する。

    :param _df: データフレーム
    :param threshold: 閾値
    :return: ohe削除後のデータフレーム
    """

    df = _df.copy()

    for column_name, item in df.iteritems():

        smaller_rate = 1

        if 2 == len(item.unique()):
            small_value = item.value_counts().sort_values().reset_index().iloc[0, 1]
            smaller_rate = small_value / len(item)

        if smaller_rate < threshold:
            print('delete column %s, delete rate %f%%' % (column_name, smaller_rate * 100))
            del df[column_name]

    print('before shape :', _df.shape)
    print('after  shape :', df.shape)

    return df


def remove_outlier(_df, _col_name, _threshold, _sign):
    """
    外れ値除去関数(データフレーム , カラム名, 閾値,　＜less or ＞more)

    :param _df: データフレーム
    :param _col_name: カラム名
    :param _threshold: 閾値
    :param _sign: 'more' or 'less'
    :return: 外れ値削除後のデータフレーム
    """

    if not isinstance(_df[_col_name][0], (int, float)):
        return _df

    print('na num: {}'.format(_df[_col_name].isna().sum()))

    if _sign == 'more':
        s_bool = _threshold < _df[_col_name]
    else:
        s_bool = _df[_col_name] < _threshold

    print('change num: {}'.format(s_bool.sum()))

    if _sign == 'more':
        # where()の仕様：指定した条件にfalseとなるものが、デフォルトではNaNとなる。
        # 指定したい場合は第二引数に指定すること。
        _df[_col_name] = _df[_col_name].where(_df[_col_name] < _threshold)

    else:
        _df[_col_name] = _df[_col_name].where(_threshold < _df[_col_name])

    print('finish: {}'.format(_col_name))

    return _df


def combine_small_categories(_df, _target_col_name, _use_categories, _other_name='OTHERS'):
    """
    対象カラム内のカテゴリのうち、レコード数の少ないカテゴリをその他の1カテゴリにまとめる。

    :param _df: データフレーム
    :param _target_col_name: 対象カラム名(カテゴリ型前提)
    :param _use_categories: 残すカテゴリ名
    :param _other_name: まとめるカテゴリに付与する名称。デフォルトは'OTHERS'
    :return:
    """

    df = _df.copy()
    useless_categories = list(set(df[_target_col_name].value_counts().index) - set(_use_categories))

    for column_name in useless_categories:
        df[_target_col_name] = df[_target_col_name].replace(column_name, _other_name)

    df[_target_col_name] = df[_target_col_name].astype('category')

    return df


def get_radian_angle(pca_result):
    """
    PCAの結果を基に、各行毎に2つの主成分ベクトルの角度を求めListにして返却する。

    :param pca_result: 2つの特徴量に集約したPCAの結果
    :return: 2つの特徴量の成す角度を格納したリスト（要素数は引数として渡されるPCAの結果に依存する）
    """
    rad = []
    for r in pca_result:
        rad.append(math.atan(r[0] / r[1]))

    return rad


def add_group_columns(_df, group_by_column, target_columns):
    df = _df.copy()

    for target in target_columns:

        if isinstance(df[target][0], (int, float, np.number)) and not isinstance(df[target][0], np.uint8):
            df[target + '_groupby_' + group_by_column + '_mean'] = df.groupby(group_by_column)[target].transform(
                np.mean)
            df[target + '_groupby_' + group_by_column + '_sum'] = df.groupby(group_by_column)[target].transform(np.sum)
            df[target + '_groupby_' + group_by_column + '_max'] = df.groupby(group_by_column)[target].transform(np.max)
            df[target + '_groupby_' + group_by_column + '_min'] = df.groupby(group_by_column)[target].transform(np.min)

        df[target + '_groupby_' + group_by_column + '_count'] = df.groupby(group_by_column)[target].transform('count')[
            10]

    return df
