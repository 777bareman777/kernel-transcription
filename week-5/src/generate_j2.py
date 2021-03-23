#!/usr/bin/env python
import argparse
from kaggler.data_io import save_data
import logging
import numpy as np
import pandas as pd
import time

from const import ID_COL, TARGET_COL


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]

    trn.drop(TARGET_COL, axis=1, inplace=True)

    # Fill Null in Age using title
    trn['Initial'] = trn.Name.str.extract('([A-Za-z]+)\.')
    tst['Initial'] = tst.Name.str.extract('([A-Za-z]+)\.')

    trn['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                            'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other',
                            'Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

    trn.loc[(trn.Age.isnull()) & (trn.Initial=='Mr'), 'Age'] = 33
    trn.loc[(trn.Age.isnull()) & (trn.Initial=='Mrs'), 'Age'] = 36
    trn.loc[(trn.Age.isnull()) & (trn.Initial=='Master'), 'Age'] = 5
    trn.loc[(trn.Age.isnull()) & (trn.Initial=='Miss'), 'Age'] = 22
    trn.loc[(trn.Age.isnull()) & (trn.Initial=='Other'), 'Age'] = 46

    tst.loc[(tst.Age.isnull()) & (tst.Initial=='Mr'), 'Age'] = 33
    tst.loc[(tst.Age.isnull()) & (tst.Initial=='Mrs'), 'Age'] = 36
    tst.loc[(tst.Age.isnull()) & (tst.Initial=='Master'), 'Age'] = 5
    tst.loc[(tst.Age.isnull()) & (tst.Initial=='Miss'), 'Age'] = 22
    tst.loc[(tst.Age.isnull()) & (tst.Initial=='Other'), 'Age'] = 46

    # Fill Null in Embarked
    trn['Embarked'].fillna('S', inplace=True)

    # add Age_band feature
    def categorize_age(x):
        if x <= 16:
            return 0
        elif x > 16 and x <= 32:
            return 1
        elif x > 32 and x <= 48:
            return 2
        elif x > 48 and x <= 64:
            return 3
        else:
            return 4

    trn['Age_band'] = trn.Age.apply(categorize_age)
    tst['Age_band'] = tst.Age.apply(categorize_age)

    # add Family_Size and Alone
    trn['Family_Size'] = 0
    trn['Family_Size'] = trn['SibSp'] + trn['Parch']
    trn['Alone'] = 0
    trn.loc[trn.Family_Size == 0, 'Alone'] = 1

    tst['Family_Size'] = 0
    tst['Family_Size'] = tst['SibSp'] + tst['Parch']
    tst['Alone'] = 0
    tst.loc[tst.Family_Size == 0, 'Alone'] = 1

    # add Fare_cat
    def categorize_fare(x):
        if x <= 7.91:
            return 0
        elif x > 7.91 and x <= 14.454:
            return 1
        elif x > 14.454 and x <= 31:
            return 2
        else:
            return 3

    trn['Fare_cat'] = trn.Fare.apply(categorize_fare)
    tst['Fare_cat'] = tst.Fare.apply(categorize_fare)

    # Change Initial, Embarked and Sex (string to numerical)
    trn.Initial = trn.Initial.map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})
    tst.Initial = tst.Initial.map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

    trn.Embarked = trn.Embarked.map({'C': 0, 'Q': 1, 'S': 2})
    tst.Embarked = tst.Embarked.map({'C': 0, 'Q': 1, 'S': 2})

    trn.Sex = trn.Sex.map({'female': 0, 'male': 1})
    tst.Sex = tst.Sex.map({'female': 0, 'male': 1})

    # drop redundant features
    trn.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)
    tst.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)

    # concat trn and tst
    df = pd.concat([trn, tst], axis=0)
    df.fillna(-1, inplace=True)

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(df.columns):
            f.write(f'{col}\n')

    logging.info('saving features')
    save_data(df.values[:n_trn, ], y, train_feature_file)
    save_data(df.values[n_trn:, ], None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_map_file)
    logging.info(f'finished ({time.time() - start:.2f} sec elasped)')
