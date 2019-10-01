import numpy as np
import pandas
import tensorflow as tf

train_csv_data_p = pandas.read_csv('./Titanic/train.csv')
test_csv_data_p = pandas.read_csv('./Titanic/test.csv')
test_csv_sub_p = pandas.read_csv('./Titanic/gender_submission.csv')

train_csv_data = train_csv_data_p.as_matrix()
test_csv_data = test_csv_data_p.as_matrix()
test_csv_sub = test_csv_sub_p.as_matrix()

# 데이터 전처리
# male -> 1
# female -> 0
for i in range(len(train_csv_data)):
    if train_csv_data[i, 4] == 'male':
        train_csv_data[i, 4] = 1
    else:  # female
        train_csv_data[i, 4] = 0

for i in range(len(test_csv_data)):
    if test_csv_data[i, 3] == 'male':
        test_csv_data[i, 3] = 1
    else:
        test_csv_data[i, 3] = 0

# 승선항 전처리
# S -> 1
# C -> 2
# Q -> 3
# nan -> 0    판단 못함, 공백
for i in range(len(train_csv_data)):
    if train_csv_data[i, 11] == 'S':
        train_csv_data[i, 11] = 1
    elif train_csv_data[i, 11] == 'C':
        train_csv_data[i, 11] = 2
    elif train_csv_data[i, 11] == 'Q':
        train_csv_data[i, 11] = 3
    if np.isnan(train_csv_data[i, 11]):
        train_csv_data[i, 11] = 0

for i in range(len(test_csv_data)):
    if test_csv_data[i, 10] == 'S':
        test_csv_data[i, 10] = 1
    elif test_csv_data[i, 10] == 'C':
        test_csv_data[i, 10] = 2
    elif test_csv_data[i, 10] == 'Q':
        test_csv_data[i, 10] = 3
    if np.isnan(test_csv_data[i, 10]):
        test_csv_data[i, 10] = 0

X_PassengerData = train_csv_data[:, [2, # Pclass
                                     4, # Sex
                                     6, # SibSp
                                     7, # Parch
                                     11 # Embarked
                                    ]]
Y_Survived = train_csv_data[:, 1:2]    # Survived
Test_X_PassengerData = test_csv_data[:, [1, # Pclass
                                         3, # Sex
                                         5, # SibSp
                                         6, # Parch
                                         10 # Embarked
                                        ]]
Test_Y_Survived = test_csv_sub[:, 1:2]    # Survived