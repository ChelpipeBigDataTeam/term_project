# coding=utf-8
"""
Описание путей к файлам данных проекта
"""
main_folder_name = "../../Data"

raw_data_1 = {
    'path': main_folder_name+'/raw/raw_data_1.xls',
    'sheets': ['2013', '2014', '2015', '2016', '2017', '369'],
    'borders': [(8, 682), (10, 469), (9, 626), (10, 652), (11, 156), (8, 968)]
}

raw_data_2 = {
    'path': main_folder_name+'/raw/raw_data_2.xlsx',
    'sheets': ['2013', '2014', '2015', '2016', '2017', '16_17'],
    'borders': [(8, 1599), (10, 2060), (10, 1136), (9, 1143), (9, 498), (6, 26)]
}


raw_data_3 = {
    'path': main_folder_name+'/raw/raw_data_3.xls',
    'sheets': ['440', '390'],
    'borders': [(6, 174), (6, 58)]
}

raw_data_4 = {
    'path': main_folder_name+'/raw/raw_data_4.xlsx',
    'sheets': ['13'],
    'borders': [(6, 450)]
}


prepared_data = {
    'full': main_folder_name+'/prepared/prepared_data.xlsx',
    '1': {
        '2013': main_folder_name+'/prepared/prepared_data_1_2013.xlsx',
        '2014': main_folder_name+'/prepared/prepared_data_1_2014.xlsx',
        '2015': main_folder_name+'/prepared/prepared_data_1_2015.xlsx',
        '2016': main_folder_name+'/prepared/prepared_data_1_2016.xlsx',
        '2017': main_folder_name+'/prepared/prepared_data_1_2017.xlsx',
        '369':  main_folder_name+'/prepared/prepared_data_1_369.xlsx'
    },
    '2': {
        '2013': main_folder_name+'/prepared/prepared_data_2_2013.xlsx',
        '2014': main_folder_name+'/prepared/prepared_data_2_2014.xlsx',
        '2015': main_folder_name+'/prepared/prepared_data_2_2015.xlsx',
        '2016': main_folder_name+'/prepared/prepared_data_2_2016.xlsx',
        '2017': main_folder_name+'/prepared/prepared_data_2_2017.xlsx',
        '16_17': main_folder_name+'/prepared/prepared_data_16_17.xlsx'
    },
    '3': {
        '440': main_folder_name+'/prepared/prepared_data_3_440.xlsx',
        '390': main_folder_name+'/prepared/prepared_data_3_390.xlsx'
    },
    '4': {
        '13': main_folder_name+'/prepared/prepared_data_4_13.xlsx'
    },
    'log': 'prepared_data.log'
}

dataset = {
    'full harding': main_folder_name+'/datasets/full_harding/full_harding_data.xlsx',
    'double harding': main_folder_name+'/datasets/double_harding/double_harding_data.xlsx'
}

raw_chemistry_data = {
    '1': main_folder_name+'/chemistry/chemistry 1.xlsx',
    # '2': '/chemistry/chemistry 2.xlsx',
    '5': main_folder_name+'/chemistry/chemistry 5.xlsx'
}

clear_chemistry_data = {
    '1_real': main_folder_name+'/chemistry/real chemistry 1.xlsx',
    '1_sertif': main_folder_name+'/chemistry/sertif chemistry 1.xlsx',
    # '2_real': main_folder_name+'/chemistry/real chemistry 2.xlsx',
    # '2_sertif': main_folder_name+'/chemistry/sertif chemistry 2.xlsx',
    '5_real': main_folder_name+'/chemistry/real chemistry 5.xlsx',
    '5_sertif': main_folder_name+'/chemistry/sertif chemistry 5.xlsx'
}

model = {
    'full harding': {
        'SVR': {
            'name': main_folder_name+"/model_data/full_harding/SVR/model.name",
            'ys_model': main_folder_name+"/model_data/full_harding/SVR/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/full_harding/SVR/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/full_harding/SVR/scaler.json",
            'titles':{
                'non_cat': main_folder_name+"/model_data/full_harding/SVR/non_cat_titles.json",
                'cat':main_folder_name + "/model_data/full_harding/SVR/cat_titles.json",
            }
        },
        'GB': {
            'name': main_folder_name+"/model_data/full_harding/GB/model.name",
            'ys_model': main_folder_name+"/model_data/full_harding/GB/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/full_harding/GB/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/full_harding/GB/scaler.json",
            'titles':{
                'non_cat': main_folder_name+"/model_data/full_harding/GB/non_cat_titles.json",
                'cat':main_folder_name + "/model_data/full_harding/GB/cat_titles.json",
            }
        },
        'LM': {
            'name': main_folder_name+"/model_data/full_harding/LM/model.name",
            'ys_model': main_folder_name+"/model_data/full_harding/LM/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/full_harding/LM/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/full_harding/LM/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/full_harding/LM/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/full_harding/LM/cat_titles.json",
            }
        },
        'NN': {
            'name': main_folder_name+"/model_data/full_harding/NN/model.name",
            'ys_model': main_folder_name+'/model_data/full_harding/NN/ys_model.json',
            'ys_weights': main_folder_name+'/model_data/full_harding/NN/ys_weights.h5',
            'tr_model': main_folder_name+'/model_data/full_harding/NN/tr_model.json',
            'tr_weights': main_folder_name+'/model_data/full_harding/NN/tr_weights.h5',
            'scaler': main_folder_name+"/model_data/full_harding/NN/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/full_harding/NN/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/full_harding/NN/cat_titles.json",
            }
        }
    },
    'double harding': {
        'SVR': {
            'name': main_folder_name+"/model_data/double_harding/SVR/model.name",
            'ys_model': main_folder_name+"/model_data/double_harding/SVR/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/double_harding/SVR/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/double_harding/SVR/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/double_harding/SVR/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/double_harding/SVR/cat_titles.json",
            }
        },
        'GB': {
            'name': main_folder_name+"/model_data/double_harding/GB/model.name",
            'ys_model': main_folder_name+"/model_data/double_harding/GB/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/double_harding/GB/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/double_harding/GB/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/double_harding/GB/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/double_harding/GB/cat_titles.json",
            }
        },
        'LM': {
            'name': main_folder_name+"/model_data/double_harding/LM/model.name",
            'ys_model': main_folder_name+"/model_data/double_harding/LM/ys_model.pkl",
            'tr_model': main_folder_name+"/model_data/double_harding/LM/tr_model.pkl",
            'scaler': main_folder_name+"/model_data/double_harding/LM/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/double_harding/LM/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/double_harding/LM/cat_titles.json",
            }
        },
        'NN': {
            'name': main_folder_name+"/model_data/double_harding/NN/model.name",
            'ys_model': main_folder_name+'/model_data/double_harding/NN/ys_model.json',
            'ys_weights': main_folder_name+'/model_data/double_harding/NN/ys_weights.h5',
            'tr_model': main_folder_name+'/model_data/double_harding/NN/tr_model.json',
            'tr_weights': main_folder_name+'/model_data/double_harding/NN/tr_weights.h5',
            'scaler': main_folder_name+"/model_data/double_harding/NN/scaler.json",
            'titles': {
                'non_cat': main_folder_name+"/model_data/double_harding/NN/non_cat_titles.json",
                'cat': main_folder_name + "/model_data/double_harding/NN/cat_titles.json",
            }
        }
    }
}

output_data = {
    'output': main_folder_name+'/output'
}
test_data = {
    'test': main_folder_name+'/test_data/test_data.xlsx',
    'input': main_folder_name+'/test_data/input_data.xlsx',
    'learn_nn': main_folder_name+'/tmp/learn_nn_test',
    'validation_res': main_folder_name+'/tmp/validation_res',
    'err_hist': main_folder_name+'/tmp/err_hist'
}

results = {
    'svr': main_folder_name+'/tmp/results/test_SVR',
    'gradient_boosting': main_folder_name+'/tmp/results/test_gradient_boosting'
}

if __name__ == '__main__':
    pass
