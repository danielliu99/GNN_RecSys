import pandas as pd
import torch
import sklearn.preprocessing as pp

train_file_path = "~/Documents/data/train.csv"
test_file_path = "~/Documents/data/test.csv"


def feature_encoding(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    '''
    label encoding 
    - Training set: label encoding fit & transform => from 0 to len(unique userId)
    - Testing set:
        - id should be in Training set
        - label encoding transform
    '''
    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()

    train_df['user_id_idx'] = le_user.fit_transform(train_df['userId'].values)
    train_df['item_id_idx'] = le_item.fit_transform(train_df['movieId'].values)

    train_user_ids = train_df['userId'].unique()
    train_item_ids = train_df['movieId'].unique()

    test_df = test_df[
        (test_df['userId'].isin(train_user_ids)) & (test_df['movieId'].isin(train_item_ids))
        ]

    test_df['user_id_idx'] = le_user.transform(test_df['userId'].values)
    test_df['item_id_idx'] = le_item.transform(test_df['movieId'].values)

    n_users = train_df['user_id_idx'].nunique()
    n_items = train_df['item_id_idx'].nunique()

    '''
    - item ids are extend after number of user, so that every NODE in the graph has a unique id.
    '''
    u_t = torch.LongTensor(train_df['user_id_idx'].values)
    i_t = torch.LongTensor(train_df['item_id_idx'].values) + n_users

    train_edge_index = torch.stack((
        torch.cat([u_t, i_t]), torch.cat([i_t, u_t])
    ))

    return train_df, test_df, n_users, n_items, u_t, i_t, train_edge_index
