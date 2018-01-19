def make_windowed(dataset, seq_length, horizon, feat_cols, resp_cols, region_col, split=0.2, predict_current=False):
    """
    This function takes data and creates a windowed dataframe to be used in time-series analysis. 

    dataset: [panda df] the raw dataframe that a time series windowed df will be created from
    seq_length: [int] the sequence length, the amount of time to be used to perform the time-series analysis
    horizon: [int] how far into the future you wish to predict
    feat_cols: [list] a list of column names that make up the feature space
    resp_cols: [list] a list of column names that make up the response
    region_col: [str] the name of the column that different time-series will be created on, i.e. different regions that contain
                independent time-series.
    split: [float] A percent split for the test set, i.e. 0.2 equals a 80/20 split for the train/test sets.
    predict_current: [bool] horizon needs to be set to 1 for this to work. This will predict at the current time
                    so if there is a sequence length of 2 instead of forecasting out the horizon length, the model
                    will predict at the current time.
    """

    if ((predict_current) and (horizon is not 1)):
        raise ValueError('If predict_current is set to True, then Horizon must be set to 1.')
    else:
        pass

    if (any(dataset[feat_cols + resp_cols].isnull().sum()) != 0):
        raise ValueError(
            'There is missing data in at least one of the columns supplied in keep_cols. Please impute this missing data as needed.')
    else:
        pass

    # check to see if there are same features in both the response and the features list
    resp_and_feats = [var for var in feat_cols if var in resp_cols]

    dense = len(resp_cols)
    regions = list(dataset[region_col].unique())
    big_dict = {}

    for i in range(len(regions)):
        big_dict.update({regions[i]: dataset[dataset[region_col] == regions[i]]})
    features = len(feat_cols)

    train_X_all, test_X_all = np.empty((0, seq_length, features)), np.empty((0, seq_length, features))
    train_y_all, test_y_all = np.empty((0, dense)), np.empty((0, dense))

    for region in regions:
        big_dict[region] = big_dict[region][resp_cols + feat_cols]
        if resp_and_feats:
            big_dict[region] = big_dict[region].loc[:, ~big_dict[region].columns.duplicated()]
        else:
            pass

        length = len(big_dict[region])
        print(length)
        test_length = int(length * split)  # 20% test set

        df_x = big_dict[region][feat_cols]
        df_y = big_dict[region][resp_cols]
        ts_x = df_x.values
        ts_y = df_y.values

        train_length = length - test_length

        train_X, train_y, test_X, test_y = [], [], [], []

        if (predict_current):
            z = 2
            q = 1
        else:
            z = 1
            q = 0

        for i in range(train_length - seq_length - horizon):
            train_X.append(ts_x[i:i + seq_length])
            train_y.append(ts_y[i + seq_length + horizon - z])

        for i in range(-test_length, 0):
            test_X.append(ts_x[i - seq_length - horizon + 1:i - horizon + 1])
            test_y.append(ts_y[i - q])

        train_X, train_y = np.array(train_X), np.array(train_y)
        test_X, test_y = np.array(test_X), np.array(test_y)

        train_X_all = np.append(train_X_all, train_X, axis=0)
        test_X_all = np.append(test_X_all, test_X, axis=0)
        train_y_all = np.append(train_y_all, train_y, axis=0)
        test_y_all = np.append(test_y_all, test_y, axis=0)

    # normalize
    mean_x = train_X_all.mean(0)
    std_x = train_X_all.std(0)

    train_X = (train_X_all - mean_x) / std_x
    test_X = (test_X_all - mean_x) / std_x
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    mean_y = train_y_all.mean(0)
    std_y = train_y_all.std(0)

    train_y = (train_y_all - mean_y) / std_y
    test_y = (test_y_all - mean_y) / std_y
    train_y = train_y.astype('float32')
    test_y = test_y.astype('float32')

    print('train_X shape:', np.shape(train_X))
    print('train_y shape:', np.shape(train_y))
    print('test_X shape:', np.shape(test_X))
    print('test_y shape:', np.shape(test_y))

    return train_X, test_X, train_y, test_y, mean_x, std_x, mean_y, std_y