# coding=utf-8
"""
Описание заголовков таблиц проекта
"""


# Соответствие между именем модели и ее ключем
target_to_key = {
    u"предел текучести": "ys_model",
    u"прочность": "tr_model"
}

target_weights_string = {
    u"предел текучести": "ys_weights",
    u"прочность": "tr_weights"
}

full_hard_title = [
    u'№ партии',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'HRB',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'Mo',
    u'Ti',
    u'Nb',
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4',
    u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    u'длина (отпуск)',
    u'длина (закалка)'
]

double_hard_title = [
    u'№ партии',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'скорость движения (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'HRB',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'Mo',
    u'Ti',
    u'Nb',
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'МКИ_1',
    u'МКИ_2',
    u'МКИ_3',
    u'МКИ_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4',
    u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    u'расход воды (МКИ)',
    u'расход воды (МКИ) (1)',
    u'расход воды (МКИ) (2)',
    u'расход воды (МКИ) (3)',
    u'длина (отпуск)',
    u'длина (закалка)'
]

not_null_double_harding = {
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'скорость движения (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'расход воды (МКИ) (1)'
}

full_hard_non_cat_title = [
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'C',
    u'Mn',
    u'Si',
    # u'P',
    # u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    # u'V',
    # u'N',
    # u'Mo',
    # u'Nb',
    # u'удельный расход воды',
    # u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    # u'длина (отпуск)',
    # u'длина (закалка)',
    # u'параметр отпуска',
    # u'параметр закалки'
]

full_hard_cat_title = [
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4'
]

double_hard_non_cat_title = [
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'скорость движения (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'C',
    u'Mn',
    u'Si',
    # u'P',
    # u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    # u'V',
    # u'N',
    # u'Mo',
    # u'Nb',
    # u'удельный расход воды',
    # u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    u'расход воды (МКИ) (1)',
    u'расход воды (МКИ) (2)',
    u'расход воды (МКИ) (3)',
    # u'длина (отпуск)',
    # u'длина (закалка)',
    #u'параметр отпуска',
    # u'параметр закалки'
]

double_hard_cat_title = [
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'МКИ_1',
    u'МКИ_2',
    u'МКИ_3',
    u'МКИ_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4'
]

target_title = [
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'HRB'
]

rus_title_1 = [
    u'дд',
    u'мм',
    u'гггг',
    u'№ партии',
    u'инд',
    u'№ ТУ',
    u'№ пп',
    u'№ инд',
    u'№ плавки',
    u'№ трубы',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'расход воды (норм.)',
    u'скорость движения (норм.)',
    u'ОКБ (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'расход воды (МКИ)',
    u'скорость движения (МКИ)',
    u'ОКБ (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'ОКБ (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'отношение тек./проч.',
    u'KCV (1)',
    u'KCV (2)',
    u'KCV (3)',
    u'В/с (1)',
    u'В/с (2)',
    u'В/с (3)',
    u'KCU',
    u'HRB',
    u'выд.',
    u'полощение',
    u'зерно',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'примечание'
]

rus_title_2 = [
    u'дд',
    u'мм',
    u'гггг',
    u'№ партии',
    u'инд',
    u'способ',
    u'№ ТУ',
    u'класс прочности',
    u'№ пакета',
    u'№ инд',
    u'№ плавки',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'расход воды (норм.)',
    u'скорость движения (норм.)',
    u'ОКБ (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'расход воды (МКИ)',
    u'скорость движения (МКИ)',
    u'ОКБ (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'ОКБ (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'отношение тек./проч.',
    u'KCV (1)',
    u'KCV (2)',
    u'KCV (3)',
    u'В/с (1)',
    u'В/с (2)',
    u'В/с (3)',
    u'KCU',
    u'HRB',
    u'выд.',
    u'полощение',
    u'зерно',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'примечание'
]

rus_title_3 = [
    u'дд',
    u'мм',
    u'гггг',
    u'№ партии',
    u'№ инд',
    u'№ плавки',
    u'№ трубы',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'расход воды (норм.)',
    u'скорость движения (норм.)',
    u'ОКБ (норм.)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'ОКБ (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'KCV (1)',
    u'KCV (2)',
    u'KCV (3)',
    u'KCV (4)',
    u'KCV (5)',
    u'KCV (6)',
    u'загиб',
    u'вел',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'Mo',
    u'Ti',
    u'Nb',
    u'N',
    u'Cэкв.',
    u'примечание'
]

rus_title_full = [
    u'дд',
    u'мм',
    u'гггг',
    u'№ партии',
    u'способ',
    u'№ инд',
    u'№ плавки',
    u'№ трубы',
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'расход воды (норм.)',
    u'скорость движения (норм.)',
    u'ОКБ (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'расход воды (МКИ)',
    u'скорость движения (МКИ)',
    u'ОКБ (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'ОКБ (отпуск)',
    u'предел текучести',
    u'прочность',
    u'удлинение',
    u'KCV (1)',
    u'KCV (2)',
    u'KCV (3)',
    u'KCV (4)',
    u'KCV (5)',
    u'KCV (6)',
    u'В/с (1)',
    u'В/с (2)',
    u'В/с (3)',
    u'KCU',
    u'HRB',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'Mo',
    u'Ti',
    u'Nb',
    u'Cэкв.',
    u'примечание'
]

chemistry_title = [
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    u'N',
    u'Mo',
    u'Ti',
    u'Nb'
]
