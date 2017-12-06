# coding=utf-8
"""
Создание данных, удобных для работы в формате pandas.DataFrame
"""

import pandas as pd
import numpy as np

import data_paths as dp
from titles import *


def create_cleaned_datafile(raw_data, title_list, sheet_name, n_begin, n_end, result_path):
    """
    Создание и сохранение данных из исходных таблиц в виде, удобном для работы в формате pandas.DataFrame
    :param raw_data: DataFrame с исходными данными
    :param title_list: Список с заголовками таблицы
    :param sheet_name: Имя листа исходного файла
    :param n_begin: Ид. начала данных (т.е. номер строчки)
    :param n_end: Ид. конца данных (т.е. номер строчки)
    :param result_path: Путь для сохранения получившегося файла
    :return: Получившийся файл в формате DataFrame
    """
    data = raw_data[sheet_name].iloc[n_begin:n_end, 0:len(title_list)]
    data = data.rename(columns=pd.Series(data=title_list, index=data.iloc[1, :].index))
    for r in xrange(n_begin, n_end):
        if pd.isnull(data.loc[r, u'№ партии']):
            data.loc[r, u'№ партии'] = data.loc[r - 1, u'№ партии']
            data.loc[r, u'темп-ра терм. (норм.)':u'ОКБ (отпуск)'] = \
                data.loc[r - 1, u'темп-ра терм. (норм.)':u'ОКБ (отпуск)']
    data[u'№ партии'] = data[u'№ партии'].apply(lambda x: unicode(x) + sheet_name)
    data.index = pd.Series(range(data.shape[0]))
    data.to_excel(result_path)
    print('Файл {} записан'.format(result_path))
    return data


full_data = []

# Загружаем первый файл
raw_data_1 = pd.read_excel(dp.raw_data_1['path'], sheetname=None)

for sheet, borders in zip(dp.raw_data_1['sheets'], dp.raw_data_1['borders']):
    df = create_cleaned_datafile(raw_data_1, rus_title_1, sheet, borders[0], borders[1], dp.prepared_data['1'][sheet])
    full_data.append(df)

# Загружаем второй файл
raw_data_2 = pd.read_excel(dp.raw_data_2['path'], sheetname=None)

for sheet, borders in zip(dp.raw_data_2['sheets'], dp.raw_data_2['borders']):
    df = create_cleaned_datafile(raw_data_2, rus_title_2, sheet, borders[0], borders[1], dp.prepared_data['2'][sheet])
    full_data.append(df)

# Загружаем третий файл
raw_data_3 = pd.read_excel(dp.raw_data_3['path'], sheetname=None)

for sheet, borders in zip(dp.raw_data_3['sheets'], dp.raw_data_3['borders']):
    df = create_cleaned_datafile(raw_data_3, rus_title_3, sheet, borders[0], borders[1], dp.prepared_data['3'][sheet])
    full_data.append(df)

# Загружаем четвертый файл
raw_data_4 = pd.read_excel(dp.raw_data_4['path'], sheetname=None)

for sheet, borders in zip(dp.raw_data_4['sheets'], dp.raw_data_4['borders']):
    df = create_cleaned_datafile(raw_data_4, rus_title_3, sheet, borders[0], borders[1], dp.prepared_data['4'][sheet])
    full_data.append(df)

# Создаем слитую таблицу
full_df = pd.DataFrame(columns=rus_title_full)

for df in full_data:
    for column in rus_title_full:
        if column not in df.columns:
            df[column] = None
    full_df = full_df.append(df[rus_title_full])

# Удалим непонятные плавки с ОЭМК, правка номера плавки
full_df = full_df[full_df[u'№ инд'] != u'ОЭМК']
full_df = full_df[full_df[u'№ инд'] != u'.-.']
full_df[u'№ плавки'] = full_df[u'№ плавки'].apply(lambda x: unicode(x).replace('.', ''))
full_df[u'№ инд'] = full_df[u'№ инд'].apply(lambda x: unicode(x).replace('.-.', '').replace('nan', ''))
full_df.index = pd.Series([i for i in range(full_df.shape[0])])

# Загружаем химию
chemistry_data = {'1_real': pd.read_excel(dp.clear_chemistry_data['1_real']),
                  '1_sertif': pd.read_excel(dp.clear_chemistry_data['1_sertif']),
                  '5_real': pd.read_excel(dp.clear_chemistry_data['5_real']),
                  '5_sertif': pd.read_excel(dp.clear_chemistry_data['5_sertif'])}

# TODO: привести в приличный вид (если не лень будет)
# Дополнение таблицы недостающими данными о хим. составе
for i in xrange(full_df.shape[0]):
    for ch_elem in chemistry_title:
        if pd.isnull(full_df.loc[i, ch_elem]):
            p_id = unicode(full_df.loc[i, u'№ инд']) + unicode(full_df.loc[i, u'№ плавки'])
            p_id = p_id.replace('nan', '')
            if not chemistry_data['1_real'][chemistry_data['1_real'][u'№ плавки'] == p_id][ch_elem].empty:
                res = chemistry_data['1_real'][chemistry_data['1_real'][u'№ плавки'] == p_id][ch_elem].iat[0]
                full_df.loc[i, ch_elem] = res
            elif not chemistry_data['5_real'][chemistry_data['5_real'][u'№ плавки'] == p_id][ch_elem].empty:
                res = chemistry_data['5_real'][chemistry_data['5_real'][u'№ плавки'] == p_id][ch_elem].iat[0]
                full_df.loc[i, ch_elem] = res
            elif not chemistry_data['1_sertif'][chemistry_data['1_sertif'][u'№ плавки'] == p_id][ch_elem].empty:
                res = chemistry_data['1_sertif'][chemistry_data['1_sertif'][u'№ плавки'] == p_id][ch_elem].iat[0]
                full_df.loc[i, ch_elem] = res
            elif not chemistry_data['5_sertif'][chemistry_data['5_sertif'][u'№ плавки'] == p_id][ch_elem].empty:
                res = chemistry_data['5_sertif'][chemistry_data['5_sertif'][u'№ плавки'] == p_id][ch_elem].iat[0]
                full_df.loc[i, ch_elem] = res

full_df.to_excel(dp.prepared_data['full'])
print('Файл {} записан'.format(dp.prepared_data['full']))
