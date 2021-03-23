#!/usr/bin/env python
import argparse
from kaggler.data_io import save_data
import logging
import numpy as np
import pandas as pd
import time
import re

from const import TARGET_COL


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_map_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file)
    tst = pd.read_csv(test_file)

    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]

    trn.drop(TARGET_COL, axis=1, inplace=True)

    # view of trn and tst
    full_data = [trn, tst]

    # Some features of my own that I have added in
    # Gives the length of the name
    for dataset in full_data:
        dataset['Name_length'] = dataset.Name.apply(len)

    # Feature that tells whether a passenger had a cabin and Titanic
    for dataset in full_data:
        dataset['Has_Cabin'] = dataset.Cabin.apply(lambda x: 0 if type(x) == float else 1)

    # Feature engineering steps taken from Sina
    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in full_data:
        dataset['FamilySize'] = dataset.SibSp + dataset.Parch + 1

    # Create new feature IsAlone from FamilySize
    for dataset in full_data:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    # Remove all NULLS in the Embarked column
    for dataset in full_data:
        dataset.Embarked = dataset.Embarked.fillna('S')

    # Remove all NULLS in the Fare column
    for dataset in full_data:
        dataset.Fare = dataset.Fare.fillna(dataset.Fare.median())

    # Remove all NULLS in the Age column
    for dataset in full_data:
        age_avg = dataset.Age.mean()
        age_std = dataset.Age.std()
        age_null_count = dataset.Age.isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

        dataset.Age[np.isnan(dataset.Age)] = age_null_random_list
        dataset.Age = dataset.Age.astype(int)

    # Define function to extract titles from passenger names
    def get_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Create a new feature Title, containing the titles of passenger names
    for dataset in full_data:
        dataset.Title = dataset.Name.apply(get_title)

    # Group all non-common titles into one single grouping 'Rare'
    for dataset in full_data:
        dataset.Title = dataset.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer',
                                               'Dona'], 'Rare')
        dataset.Title = dataset.Title.replace('Mlle', 'Miss')
        dataset.Title = dataset.Title.replace('Ms', 'Miss')
        dataset.Title = dataset.Title.replace('Mme', 'Mrs')

    for dataset in full_data:
        # Mapping Sex
        dataset.Sex = dataset.Sex.map({'female': 0, 'male': 1}).astype(int)

        # Mapping Titles
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
        dataset.Title = dataset.Title.map(title_mapping)
        dataset.Title = dataset.Title.fillna(0)

        # Mapping Embarked
        dataset.Embarked = dataset.Embarked.map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Mapping Fare
        def map_fare(x):
            if x <= 7.91:
                return 0
            elif x <= 14.454:
                return 1
            elif x <= 31:
                return 2
            else:
                return 3

        dataset.Fare = dataset.Fare.apply(map_fare)
        dataset.Fare = dataset.Fare.astype(int)
    
        # Mapping Age
        def map_age(x):
            if x <= 16:
                return 0
            elif x <= 32:
                return 1
            elif x <= 48:
                return 2
            elif x <= 64:
                return 3
            else:
                return 4

        dataset.Age = dataset.Age.apply(map_age)

    # drop redundant features
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    trn.drop(drop_elements, axis=1, inplace=True)
    tst.drop(drop_elements, axis=1, inplace=True)

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
