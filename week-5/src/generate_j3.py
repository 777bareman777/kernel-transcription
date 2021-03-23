#!/usr/bin/env python
import argparse
from kaggler.data_io import save_data
import logging
import numpy as np
import pandas as pd
import time

from const import TARGET_COL


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]

    trn.drop(TARGET_COL, axis=1, inplace=True)

    # Fill empty and NaNs values with NaN
    trn = trn.fillna(np.nan)
    tst = tst.fillna(np.nan)

    # Fill Null in Fare about test
    tst.Fare.fillna(tst.Fare.median())

    # Apply log to Fare to reduce skewneww distribution
    trn.Fare = trn.Fare.map(lambda i: np.log(i) if i > 0 else 0)
    tst.Fare = tst.Fare.map(lambda i: np.log(i) if i > 0 else 0)

    # Fill Null in Embarked about train
    trn.Embarked.fillna('S')

    # convert Embarked into categorical value
    trn.Embarked = trn.Embarked.map({'S': 0, 'Q': 1, 'C': 2})
    tst.Embarked = tst.Embarked.map({'S': 0, 'Q': 1, 'C': 2})

    # convert Sex into categorical value 0 for male and 1 for female
    trn.Sex = trn.Sex.map({'male': 0, 'female': 1})
    tst.Sex = tst.Sex.map({'male': 0, 'female': 1})

    ## Fill Age with the median age of similar rows
    ## according to Pclass, Parch and SibSp 

    # Index of NaN age rows about train
    train_index_nan_age = list(trn.Age[trn.Age.isnull()].index)
    for i in train_index_nan_age:
        age_med = trn.Age.median()
        age_pred = trn.Age[(trn.SibSp == trn.iloc[i].SibSp) &
                           (trn.Parch == trn.iloc[i].Parch) &
                           (trn.Pclass == trn.iloc[i].Pclass)].median()
        if not np.isnan(age_pred):
            trn.Age.iat[i] = age_pred
        else:
            trn.Age.iat[i] = age_med

    test_index_nan_age = list(tst.Age[tst.Age.isnull()].index)
    for i in test_index_nan_age:
        age_med = tst.Age.median()
        age_pred = tst.Age[(tst.SibSp == tst.iloc[i].SibSp) &
                           (tst.Parch == tst.iloc[i].Parch) &
                           (tst.Pclass == tst.iloc[i].Pclass)].median()
        if not np.isnan(age_pred):
            tst.Age.iat[i] = age_pred
        else:
            tst.Age.iat[i] = age_med

    # Get title from Name
    trn_title = [i.split(',')[1].split('.')[0].strip() for i in trn.Name]
    trn['Title'] = pd.Series(trn_title)
    tst_title = [i.split(',')[1].split('.')[0].strip() for i in tst.Name]
    tst['Title'] = pd.Series(tst_title)

    # Convert to categorical values Title
    trn.Title = trn.Title.replace(['Lady', 'the Countess', 'Countess',
                                   'Capt', 'Col', 'Don', 'Dr', 'Major',
                                   'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    trn.Title = trn.Title.map({'Master': 0, 'Miss': 1, 'Ms': 1, 'Mme': 1,
                               'Mlle': 1, 'Mrs': 1, 'Mr': 2, 'Rare': 3})
    trn.Title = trn.Title.astype(int)

    tst.Title = tst.Title.replace(['Lady', 'the Countess', 'Countess',
                                   'Capt', 'Col', 'Don', 'Dr', 'Major',
                                   'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    tst.Title = tst.Title.map({'Master': 0, 'Miss': 1, 'Ms': 1, 'Mme': 1,
                               'Mlle': 1, 'Mrs': 1, 'Mr': 2, 'Rare': 3})
    tst.Title = tst.Title.astype(int)

    # Create a family size descriptor from SibSp and Parch
    trn['Fsize'] = trn.SibSp + trn.Parch + 1
    tst['Fsize'] = tst.SibSp + tst.Parch + 1

    # Create new feature of family size
    trn['Single'] = trn.Fsize.map(lambda s: 1 if s == 1 else 0)
    trn['SmallF'] = trn.Fsize.map(lambda s: 1 if s == 2 else 0)
    trn['MedF'] = trn.Fsize.map(lambda s: 1 if 3 <= s <= 4 else 0)
    trn['LargeF'] = trn.Fsize.map(lambda s: 1 if s >= 5 else 0)

    tst['Single'] = tst.Fsize.map(lambda s: 1 if s == 1 else 0)
    tst['SmallF'] = tst.Fsize.map(lambda s: 1 if s == 2 else 0)
    tst['MedF'] = tst.Fsize.map(lambda s: 1 if 3 <= s <= 4 else 0)
    tst['LargeF'] = tst.Fsize.map(lambda s: 1 if s >= 5 else 0)

    # convert to indicator values Title and Embarked
    trn = pd.get_dummies(trn, columns=['Title'])
    tst = pd.get_dummies(tst, columns=['Embarked'], prefix='Em')

    # Replace the Cabin number by the type of cabin 'X' if not
    trn.Cabin = pd.Series(['X' if pd.isnull(i) else i[0] for i in trn.Cabin])
    tst.Cabin = pd.Series(['X' if pd.isnull(i) else i[0] for i in tst.Cabin])

    # convert to indicator values Cabin
    trn = pd.get_dummies(trn, columns=['Cabin'], prefix='Cabin')
    tst = pd.get_dummies(tst, columns=['Cabin'], prefix='Cabin')

    # Treat Ticket by extracting the ticket prefix.
    # When there is no prefix it returns X.
    trn_ticket = []
    for i in list(trn.Ticket):
        if i.isdigit():
            trn_ticket.append('X')
        else:
            trn_ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])
    trn.Ticket = trn_ticket
    trn = pd.get_dummies(trn, columns=['Ticket'], prefix='T')

    tst_ticket = []
    for i in list(tst.Ticket):
        if i.isdigit():
            tst_ticket.append('X')
        else:
            tst_ticket.append(i.replace('.', '').replace('/', '').strip().split(' ')[0])
    tst.Ticket = tst_ticket
    tst = pd.get_dummies(tst, columns=['Ticket'], prefix='T')

    # Create categorical valeus for Pclass
    trn.Pclass = trn.Pclass.astype('category')
    trn = pd.get_dummies(trn, columns=['Pclass'], prefix='Pc')

    tst.Pclass = tst.Pclass.astype('category')
    tst = pd.get_dummies(tst, columns=['Pclass'], prefix='Pc')

    # drop redundant features
    trn.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    tst.drop(['Name', 'PassengerId'], axis=1, inplace=True)

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
