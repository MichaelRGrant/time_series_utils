import numpy as np


class Window:
    """
    This function takes data and creates a windowed dataframe to be used in time-series analysis. 

    dataset: [panda df] the raw dataframe that a time series windowed df will be created from
    seq_length: [int] the sequence length, the amount of time to be used to perform the 
                time-series analysis
    horizon: [int] how far into the future you wish to predict
    feat_cols: [list] a list of column names that make up the feature space
    resp_cols: [list] a list of column names that make up the response
    region_col: [str] the name of the column that different time-series will be created on, i.e. 
                different regions that contain
                independent time-series.
    split: [float] A percent split for the test set, i.e. 0.2 equals a 80/20 split for the 
                train/test sets.
    resp_width: [int] If you want to predict out to a set distance you set the horizon to that 
                    time point and this value to 0, however if you want to predict every value 
                    between let's say now and some point in the future you set horizon to 1 and
                    the resp_width to that point. The algorithm will then predict every time point.
    predict_current: [bool] horizon needs to be set to 1 for this to work. This will predict at 
                    the current time so if there is a sequence length of 2 instead of forecasting 
                    out the horizon length, the model
                    will predict at the current time.
    """

    def __init__(self, seq_length, horizon, feat_cols, resp_cols, region_col, resp_width, predict_current=False):
        self.seq_length = seq_length
        self.horizon = horizon
        self.feat_cols = feat_cols
        self.resp_cols = resp_cols
        self.region_col = region_col
        self.resp_width = resp_width
        self.predict_current = predict_current
        self.response = len(resp_cols)

        if self.predict_current and self.horizon is not 1:
            raise ValueError('If predict_current is set to True, then Horizon must be set to 1.')
        else:
            pass

    def make(self, dataset, split=0.2):
        # check to see if there are same features in both the response and the features list
        resp_and_feats = [var for var in self.feat_cols if var in self.resp_cols]
        dense = len(self.resp_cols)

        if any(dataset[self.feat_cols + self.resp_cols].isnull().sum()) != 0:
            raise ValueError(
                'There is missing data in at least one of the columns supplied in keep_cols. Please impute this '
                'missing data as needed.')
        else:
            pass

        regions = list(dataset[self.region_col].unique())
        big_dict = {}

        for i in range(len(regions)):
            big_dict.update({regions[i]: dataset[dataset[self.region_col] == regions[i]]})

        features = len(self.feat_cols)

        if self.resp_width == 0:
            train_X_all, test_X_all = np.empty((0, self.seq_length, features)), \
                                      np.empty((0, self.seq_length, features))
            train_y_all, test_y_all = np.empty((0, self.response)), \
                                      np.empty((0, self.response))
        else:
            train_X_all, test_X_all = np.empty((0, self.seq_length, features)), \
                                      np.empty((0, self.seq_length, features))
            train_y_all, test_y_all = np.empty((0, self.resp_width, self.response)), \
                                      np.empty((0, self.resp_width, self.response))

        for region in regions:
            big_dict[region] = big_dict[region][self.resp_cols + self.feat_cols]
            if resp_and_feats:
                big_dict[region] = big_dict[region].loc[:, ~big_dict[region].columns.duplicated()]
            else:
                pass

            length = len(big_dict[region])
            if split:
                test_length = int(length * split)
            else:
                pass

            df_x = big_dict[region][self.feat_cols]
            df_y = big_dict[region][self.resp_cols]
            ts_x = df_x.values
            ts_y = df_y.values

            train_length = length - test_length

            train_X, train_y, test_X, test_y = [], [], [], []

            if self.predict_current:
                z = 2
                q = 1
            else:
                z = 1
                q = 0

            if self.resp_width != 0:
                for i in range(train_length - self.seq_length - self.horizon - self.resp_width):
                    train_X.append(ts_x[i:i + self.seq_length])
                    train_y.append(
                        ts_y[i + self.seq_length + self.horizon - z:i + self.seq_length + self.horizon - z + self.resp_width])
                for i in range(-test_length, -self.resp_width):
                    test_X.append(ts_x[i - self.seq_length - self.horizon + 1:i - self.horizon + 1])
                    test_y.append(ts_y[i - q:i - q + (self.resp_width)])

            else:
                for i in range(train_length - self.seq_length - self.horizon):
                    train_X.append(ts_x[i:i + self.seq_length])
                    train_y.append(ts_y[i + self.seq_length + self.horizon - z])
                for i in range(-test_length, 0):
                    test_X.append(ts_x[i - self.seq_length - self.horizon + 1:i - self.horizon + 1])
                    test_y.append(ts_y[i - q])

            train_X, train_y = np.array(train_X), np.array(train_y)
            test_X, test_y = np.array(test_X), np.array(test_y)

            train_X_all = np.append(train_X_all, train_X, axis=0)
            test_X_all = np.append(test_X_all, test_X, axis=0)
            train_y_all = np.append(train_y_all, train_y, axis=0)
            test_y_all = np.append(test_y_all, test_y, axis=0)

        train_X, test_X, train_y, test_y, mean_y, std_y = self.center_scale(train_X=train_X_all, test_X=test_X_all,
                                                                            train_y=train_y_all, test_y=test_y_all)

        return train_X, test_X, train_y, test_y, mean_y, std_y

    def center_scale(self, train_X, test_X, train_y, test_y):

        mean_x = train_X.mean(0)
        std_x = train_X.std(0)

        train_X_norm = (train_X - mean_x) / std_x
        test_X_norm = (test_X - mean_x) / std_x
        train_X_norm = train_X_norm.astype('float32')
        test_X_norm = test_X_norm.astype('float32')

        mean_y = train_y.mean(0)
        std_y = train_y.std(0)

        train_y_norm = (train_y - mean_y) / std_y
        test_y_norm = (test_y - mean_y) / std_y
        train_y_norm = train_y_norm.astype('float32')
        test_y_norm = test_y_norm.astype('float32')

        if self.resp_width != 0:
            std_y = std_y.ravel()
            mean_y = mean_y.ravel()
            train_y_norm = train_y_norm.reshape(train_y_norm.shape[0], train_y_norm.shape[1] * self.response)
            test_y_norm = test_y_norm.reshape(test_y_norm.shape[0], test_y_norm.shape[1] * self.response)
        else:
            pass

        print('train_X shape:', np.shape(train_X_norm))
        print('train_y shape:', np.shape(train_y_norm))
        print('test_X shape:', np.shape(test_X_norm))
        print('test_y shape:', np.shape(test_y_norm))

        return train_X_norm, test_X_norm, train_y_norm, test_y_norm, mean_y, std_y
