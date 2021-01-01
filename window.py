import itertools
from typing import Optional

import numpy as np
import pandas as pd
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook


class Window:
    """
    This class takes data and creates a windowed dataframe to be used for time-series DL modeling.

    Parameters:
    -------
    dataset: pd.DataFrame
        The raw dataframe that a time series windowed df will be created from
    seq_length: int
        the sequence length, the amount of time to be used to perform the time-series analysis
    horizon: int
        how far into the future you wish to predict
    feat_cols: list
        a list of column names that make up the feature space
    resp_cols: list
        a list of column names that make up the response
    group_col: str
        the name of the column that different time-series will be created on, i.e.
        different regions, stocks, people, groups etc., that contain independent time-series.
    split: float
        A percent split for the test set, i.e. 0.2 equals a 80/20 split for the
        train/test sets. If no split is required and a windowed df of the entire data is to
        be created, set split to None.
    resp_width: int
        If you want to predict out to a set distance you set the horizon to that
        time point and this value to 0, however if you want to predict every value
        between let's say now and some point in the future you set horizon to 1 and
        the resp_width to that point. The algorithm will then predict every time point.
    mean/std: float
        When making a windowed df for testing only, you normalize using the mean and std
        deviation set from the training set. These values must be known before hand and
        set here. The test set will be normalized using these values.
    predict_current: bool
        horizon needs to be set to 1 for this to work. This will predict at
        the current time so if there is a sequence length of 2 instead of forecasting
        out the horizon length, the model will predict at the current time.
    save_normal_params: bool
        save the normalization parameters for future predictions
    test_set: bool
        set this to true if building a window dataframe for testing purposes only. This will require
        manual setting of the mean and standard deviations generated from training.
    """

    def __init__(
        self,
        seq_length,
        horizon,
        feat_cols,
        resp_cols,
        group_col,
        resp_width,
        predict_current=False,
        classification=False,
        scale_by_group=False,
    ):
        self.seq_length = seq_length
        self.horizon = horizon
        self.feat_cols = feat_cols
        self.resp_cols = resp_cols
        self.group_col = group_col
        self.resp_width = resp_width
        self.predict_current = predict_current
        self.response = len(resp_cols)
        self.classification = classification
        self.scale_by_group = scale_by_group
        self.norm_params = {}

        if self.predict_current and self.horizon != 1:
            raise ValueError(
                "If predict_current is set to True, then Horizon must be set to 1."
            )
        if self.classification and self.resp_width != 0:
            raise ValueError(
                f"`classification` == True, therefore `resp_width` must be 0; currently it is {self.resp_width}"
            )

    # TODO add encoder function to all paths in make function for classification scheme
    # TODO scale_by_group is currently not available if using `split`

    def make(
        self,
        dataset: pd.DataFrame,
        split: Optional[int] = 0.2,
        norm_params: Optional[EasyDict] = None,
        test_set: bool = False,
        scale: str = "minmax",
        min_val: int = 0,
        max_val: int = 1,
    ) -> EasyDict:
        """
        This method creates the actual training and/or testing sets of data as an EasyDict.
        Both the X and y datasets are created.

        Parameters:
        -------
        dataset:
            The data to be used to create the windowed dataframe.
        split:
            This is optional either a float for the fraction of the data to be the test set
            or None if the data is manually split beforehand.
        norm_params:
            If test_set is True then norm_params from the training set will be required.
        test_set:
            Set to true if this is a test/validation set.
        scale:
            Type of scaling to take place: one of {"standard", "minmax"}
        min_val:
            Used for minmax, the min value
        max_val:
            Used for minmax, the max value

        Returns:
        -------
        EasyDict
        """

        # Check for inf values
        if any(np.isinf(dataset[self.feat_cols]).sum()) != 0:
            raise ValueError(
                "There are some `inf` values in at least one of the feature columns in the data. "
            )
        # Check for missing data.
        if any(dataset[self.feat_cols + self.resp_cols].isnull().sum()) != 0:
            raise ValueError(
                "There is missing data in at least one of the columns supplied in keep_cols. Please impute this "
                "missing data as needed."
            )
        if test_set and norm_params is None:
            raise ValueError(
                "If `test_set` is True then `norm_params` must not be None."
            )

        # check to see if there are same features in both the response and the features list
        resp_and_feats = [var for var in self.feat_cols if var in self.resp_cols]
        # dense = len(self.resp_cols)

        num_features = len(self.feat_cols)
        if self.resp_width == 0:
            train_X_all, test_X_all = (
                np.empty((0, self.seq_length, num_features)),
                np.empty((0, self.seq_length, num_features)),
            )
            train_y_all, test_y_all = (
                np.empty((0, self.response)),
                np.empty((0, self.response)),
            )
        else:
            train_X_all, test_X_all = (
                np.empty((0, self.seq_length, num_features)),
                np.empty((0, self.seq_length, num_features)),
            )
            train_y_all, test_y_all = (
                np.empty((0, self.resp_width, self.response)),
                np.empty((0, self.resp_width, self.response)),
            )

        group_dict, error_regions, end_idx = {}, [], []
        train_X_all_list, train_y_all_list, test_X_all_list, test_y_all_list = (
            [],
            [],
            [],
            [],
        )

        groups = sorted(dataset[self.group_col].unique())
        for group in tqdm_notebook(groups, total=len(groups)):
            group_dict[group] = dataset.query(f"{self.group_col} == @group")[
                self.resp_cols + self.feat_cols
            ]
            if resp_and_feats:
                # remove duplicate columns??
                group_dict[group] = group_dict[group].loc[
                    :, ~group_dict[group].columns.duplicated()
                ]

            length = group_dict[group].shape[0]
            if split:
                test_length = int(length * split)
                train_length = length - test_length
            else:
                train_length = length

            df_x = group_dict[group][self.feat_cols]
            df_y = group_dict[group][self.resp_cols]

            ts_x = df_x.values
            ts_y = df_y.values

            train_X, train_y, test_X, test_y = [], [], [], []

            if self.predict_current:
                z = 2
                q = 1
            else:
                z = 1
                q = 0

            if self.resp_width != 0:
                for i in range(
                    train_length - self.seq_length - self.horizon - self.resp_width
                ):
                    train_X.append(ts_x[i : i + self.seq_length])
                    train_y.append(
                        ts_y[
                            i
                            + self.seq_length
                            + self.horizon
                            - z : i
                            + self.seq_length
                            + self.horizon
                            - z
                            + self.resp_width
                        ]
                    )
                end_idx.append(i + self.seq_length)

                if split:
                    for i in range(-test_length, -self.resp_width):
                        test_X.append(
                            ts_x[
                                i
                                - self.seq_length
                                - self.horizon
                                + 1 : i
                                - self.horizon
                                + 1
                            ]
                        )
                        test_y.append(ts_y[i - q : i - q + self.resp_width])

            else:
                for i in range(train_length - self.seq_length - self.horizon):
                    train_X.append(ts_x[i : i + self.seq_length])
                    train_y.append(ts_y[i + self.seq_length + self.horizon - z])
                end_idx.append(i)
                if split:
                    for i in range(-test_length, 0):
                        test_X.append(
                            ts_x[
                                i
                                - self.seq_length
                                - self.horizon
                                + 1 : i
                                - self.horizon
                                + 1
                            ]
                        )
                        test_y.append(ts_y[i - q])

            if split:
                try:
                    train_X, train_y = np.array(train_X), np.array(train_y)
                    test_X, test_y = np.array(test_X), np.array(test_y)

                    train_X_all_list.append(train_X)
                    test_X_all_list.append(test_X)
                    train_y_all_list.append(train_y)
                    test_y_all_list.append(test_y)
                except:
                    error_regions.append(group)

            else:
                train_X, train_y = np.array(train_X), np.array(train_y)
                train_X_all_list.append(train_X)
                train_y_all_list.append(train_y)

        end_idx = list(zip(groups, end_idx))

        if not self.scale_by_group:
            train_X_all = np.concatenate(train_X_all_list, axis=0)
            train_y_all = np.concatenate(train_y_all_list, axis=0)
            if split:
                test_X_all = np.concatenate(test_X_all_list, axis=0)
                test_y_all = np.concatenate(test_y_all_list, axis=0)

        if error_regions:
            print(error_regions)

        if scale == "standard":
            if split:
                train_test_data = self.center_scale(
                    train_X=train_X_all,
                    test_X=test_X_all,
                    train_y=train_y_all,
                    test_y=test_y_all,
                    scale_type="both",
                )
                train_test_data.current = self.predict_current
                group_idx = make_start_end_index_dict(end_idx)
                train_test_data.group_idx = group_idx
                return train_test_data

            else:
                if test_set:
                    # The default to save a dataset is `train` so when the test set is being used here,
                    # it is saved as train_X_all and then saved into the test_X method argument.
                    train_test_data = self.center_scale(
                        test_X=train_X_all,
                        test_y=train_y_all,
                        scale_type="testing_only",
                        norm_params=norm_params,
                    )
                    train_test_data.current = self.predict_current
                    return train_test_data

                else:
                    train_test_data = self.center_scale(
                        train_X=train_X_all,
                        train_y=train_y_all,
                        scale_type="training_only",
                    )
                    train_test_data.current = self.predict_current
                    group_idx = make_start_end_index_dict(end_idx)
                    train_test_data.group_idx = group_idx

                    return train_test_data

        elif scale == "minmax":
            if split:
                train_test_data = self.minmax(
                    train_X=train_X_all,
                    test_X=test_X_all,
                    train_y=train_y_all,
                    test_y=test_y_all,
                    min_val=0,
                    max_val=1,
                    scale_type="both",
                )

                train_test_data.current = self.predict_current
                group_idx = make_start_end_index_dict(end_idx)
                train_test_data.group_idx = group_idx

                if self.classification:
                    encoder = LabelEncoder()
                    train_y = encoder.fit_transform(train_y)
                    test_y = encoder.fit_transform(test_y)
                    train_test_data.train_y = train_y
                    train_test_data.test_y = test_y
                return train_test_data

            else:
                if test_set:
                    if self.scale_by_group:
                        minmax_output_list = [
                            self.minmax(
                                test_X=X,
                                test_y=y,
                                min_val=0,
                                max_val=1,
                                norm_params=norm_params,
                                group=group,
                                scale_type="testing_only",
                            )
                            for X, y, group in zip(
                                train_X_all_list, train_y_all_list, groups
                            )
                        ]
                        train_test_data = EasyDict(
                            {
                                "test_X": np.concatenate(
                                    [x["test_X"] for x in minmax_output_list], axis=0
                                ),
                                "test_y": np.concatenate(
                                    [x["test_y"] for x in minmax_output_list], axis=0
                                ),
                            }
                        )

                    else:
                        train_test_data = self.minmax(
                            test_X=train_X_all,
                            test_y=train_y_all,
                            norm_params=norm_params,
                            min_val=min_val,
                            max_val=max_val,
                            scale_type="testing_only",
                        )
                    train_test_data.current = self.predict_current
                    group_idx = make_start_end_index_dict(end_idx)
                    train_test_data.group_idx = group_idx
                    return train_test_data
                else:
                    if self.scale_by_group:
                        minmax_output_list = [
                            self.minmax(
                                train_X=X,
                                train_y=y,
                                min_val=0,
                                max_val=1,
                                scale_type="training_only",
                                group=group,
                            )
                            for X, y, group in zip(
                                train_X_all_list, train_y_all_list, groups
                            )
                        ]
                        train_test_data = EasyDict(
                            {
                                "train_X": np.concatenate(
                                    [x["train_X"] for x in minmax_output_list], axis=0
                                ),
                                "train_y": np.concatenate(
                                    [x["train_y"] for x in minmax_output_list], axis=0
                                ),
                            }
                        )

                    else:
                        train_test_data = self.minmax(
                            train_X=train_X_all,
                            train_y=train_y_all,
                            min_val=min_val,
                            max_val=max_val,
                            scale_type="training_only",
                        )
                    train_test_data.current = self.predict_current
                    group_idx = make_start_end_index_dict(end_idx)
                    train_test_data.group_idx = group_idx
                    return train_test_data
        else:
            return train_X_all_list, train_y_all_list, groups

    # TODO add classification to this function
    def center_scale(
        self,
        train_X: Optional[np.array] = None,
        train_y: Optional[np.array] = None,
        test_X: Optional[np.array] = None,
        test_y: Optional[np.array] = None,
        norm_params: Optional[EasyDict] = None,
        scale_type: str = "both",
    ) -> EasyDict:
        """
        Take the data and normalizes by center and scale.

        Parameters:
        -------
        train_X: np.array
        train_y: np.array
        test_X: np.array
        test_y: np.array
        norm_params: EasyDict
            The parameters used to normalize. These are obtained from the training set and used
            to normalize the testing set.
        scale_type: str
            This dictates what type of normalization that should take place.
                "both": If the data has been split previously, then input both the
                         training and testing sets.
                "training_only": This will normalize the training set only. The normalization
                                 parameters will be saved in the class object.
                "testing_only": This will normalize the test set and requires the input of
                                normalization parameters from the training set.

        Returns:
        -------
        Tuple
            Returns EasyDicts for the train/test data and the normalizing parameters.
        """
        if scale_type not in ["both", "training_only", "testing_only"]:
            raise ValueError(
                "`scale_type` must be one of {'both', 'training_only', 'testing_only'}"
            )

        if scale_type == "testing_only" and norm_params is None:
            raise ValueError(
                "if `scale_type` is set to 'testing_only', `norm_params` must not be None."
            )

        if scale_type in ["both", "training_only"]:
            mean_x = train_X.mean(0)
            std_x = train_X.std(0)
            mean_y = train_y.mean(0)
            std_y = train_y.std(0)

            normalizing_params = {
                "mean_y": mean_y,
                "std_y": std_y,
                "mean_x": mean_x,
                "std_x": std_x,
            }

            self.norm_params = EasyDict(normalizing_params)

            train_X_norm = ((train_X - mean_x) / std_x).astype("float32")
            train_y_norm = ((train_y - mean_y) / std_y).astype("float32")

            if scale_type == "both":
                if test_X is None or test_y is None:
                    raise ValueError(
                        "For `scale_type` == 'both', the function must include the split test set."
                    )

                test_X_norm = ((test_X - mean_x) / std_x).astype("float32")
                test_y_norm = ((test_y - mean_y) / std_y).astype("float32")

                if self.resp_width != 0:
                    std_y = std_y.ravel()
                    mean_y = mean_y.ravel()
                    train_y_norm = train_y_norm.reshape(
                        train_y_norm.shape[0], train_y_norm.shape[1] * self.response
                    )
                    test_y_norm = test_y_norm.reshape(
                        test_y_norm.shape[0], test_y_norm.shape[1] * self.response
                    )
                    normalizing_params["std_y"] = std_y
                    normalizing_params["mean_y"] = mean_y
                    self.norm_params = EasyDict(normalizing_params)

                train_test_dict = {
                    "train_X": train_X_norm,
                    "test_X": test_X_norm,
                    "train_y": train_y_norm,
                    "test_y": test_y_norm,
                }

                return EasyDict(train_test_dict)

            elif scale_type == "training_only":
                if self.resp_width != 0:
                    std_y = std_y.ravel()
                    mean_y = mean_y.ravel()
                    train_y_norm = train_y_norm.reshape(
                        train_y_norm.shape[0], train_y_norm.shape[1] * self.response
                    )
                    normalizing_params["std_y"] = std_y
                    normalizing_params["mean_y"] = mean_y
                    self.norm_params = EasyDict(normalizing_params)

                train_test_dict = {
                    "train_X": train_X_norm,
                    "train_y": train_y_norm,
                }

                return EasyDict(train_test_dict)

        elif scale_type == "testing_only":
            if test_X is None or test_y is None:
                raise ValueError(
                    "For `scale_type` == 'testing_only', the function must include the split test set."
                )
            test_X_norm = ((test_X - norm_params.mean_x) / norm_params.std_x).astype(
                "float32"
            )
            test_y_norm = ((test_y - norm_params.mean_y) / norm_params.std_y).astype(
                "float32"
            )
            if self.resp_width != 0:
                test_y_norm = test_y_norm.reshape(
                    test_y_norm.shape[0], test_y_norm.shape[1] * self.response
                )
                # test_y_norm = test_y_norm.reshape(test_y_norm.shape[0], test_y_norm.shape[1] * self.response)

            train_test_dict = {"test_X": test_X_norm, "test_y": test_y_norm}

            return EasyDict(train_test_dict)

    def minmax(
        self,
        min_val: int,
        max_val: int,
        train_X: Optional[np.array] = None,
        train_y: Optional[np.array] = None,
        test_X: Optional[np.array] = None,
        test_y: Optional[np.array] = None,
        norm_params: Optional[EasyDict] = None,
        scale_type: str = "both",
        group: Optional[str] = None,
    ):
        """
        Take the data and normalizes by minmax.

        Parameters:
        -------
        min_val: int
            The minimum value
        max_val: int
            The maximum value
        train_X: np.array
        train_y: np.array
        test_X: np.array
        test_y: np.array
        norm_params: EasyDict
            The parameters used to normalize. These are obtained from the training set and used
            to normalize the testing set.
        scale_type: str
            This dictates what type of normalization that should take place.
                "both": If the data has been split previously, then input both the
                         training and testing sets.
                "training_only": This will normalize the training set only. The normalization
                                 parameters will be saved in the class object.
                "testing_only": This will normalize the test set and requires the input of
                                normalization parameters from the training set.

        Returns:
        -------
        Tuple
            Returns EasyDicts for the train/test data and the normalizing parameters.
        """
        if scale_type not in ["both", "training_only", "testing_only"]:
            raise ValueError(
                "`scale_type` must be one of {'both', 'training_only', 'testing_only'}"
            )
        if scale_type == "testing_only" and norm_params is None:
            raise ValueError(
                "if `scale_type` is set to 'testing_only', `norm_params` must not be None."
            )

        if scale_type in ["both", "training_only"]:
            min_x = train_X.min(0)
            max_x = train_X.max(0)
            if self.classification:
                min_y = 0
                max_y = 1
            else:
                min_y = train_y.min(0)
                max_y = train_y.max(0)

            normalizing_params = {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            }
            if self.scale_by_group:
                self.norm_params[group] = EasyDict(normalizing_params)
            else:
                self.norm_params["all"] = EasyDict(normalizing_params)

            train_X_norm = (
                (((max_val - min_val) * (train_X - min_x)) / (max_x - min_x)) + min_val
            ).astype("float32")
            train_y_norm = (
                ((max_val - min_val) * (train_y - min_y)) / (max_y - min_y)
            ) + min_val

            if self.classification:
                train_y_norm = train_y_norm.astype("int32")
            else:
                train_y_norm = train_y_norm.astype("float32")

            if scale_type == "both":
                if test_X is None or test_y is None:
                    raise ValueError(
                        "For `scale_type` == 'both', the function must include the split test set."
                    )
                test_X_norm = (
                    (((max_val - min_val) * (test_X - min_x)) / (max_x - min_x))
                    + min_val
                ).astype("float32")
                test_y_norm = (
                    ((max_val - min_val) * (test_y - min_y)) / (max_y - min_y)
                ) + min_val
                if self.classification:
                    test_y_norm = test_y_norm.astype("int32")
                else:
                    test_y_norm = test_y_norm.astype("float32")

                if ~self.classification and self.resp_width != 0:
                    min_y = min_y.ravel()
                    max_y = max_y.ravel()
                    train_y_norm = train_y_norm.reshape(
                        train_y_norm.shape[0], train_y_norm.shape[1] * self.response
                    )
                    test_y_norm = test_y_norm.reshape(
                        test_y_norm.shape[0], test_y_norm.shape[1] * self.response
                    )
                    normalizing_params["min_y"] = min_y
                    normalizing_params["max_y"] = max_y
                    self.norm_params = EasyDict(normalizing_params)

                train_test_dict = {
                    "train_X": train_X_norm,
                    "test_X": test_X_norm,
                    "train_y": train_y_norm,
                    "test_y": test_y_norm,
                }

                return EasyDict(train_test_dict)

            elif scale_type == "training_only":
                if ~self.classification and self.resp_width != 0:
                    min_y = min_y.ravel()
                    max_y = max_y.ravel()
                    train_y_norm = train_y_norm.reshape(
                        train_y_norm.shape[0], train_y_norm.shape[1] * self.response
                    )
                    normalizing_params["min_y"] = min_y
                    normalizing_params["max_y"] = max_y
                    self.norm_params = EasyDict(normalizing_params)

                train_test_dict = {
                    "train_X": train_X_norm,
                    "train_y": train_y_norm,
                }

                return EasyDict(train_test_dict)

        elif scale_type == "testing_only":
            if test_X is None or test_y is None:
                raise ValueError(
                    "For `scale_type` == 'testing_only', the function must include the split test set."
                )
            if norm_params is None:
                raise ValueError(
                    "For `scale_type` == 'testing_only' `norm_params` must not be None."
                )
            if self.scale_by_group:
                test_X_norm = (
                    (
                        ((max_val - min_val) * (test_X - norm_params[group].min_x))
                        / (norm_params[group].max_x - norm_params[group].min_x)
                    )
                    + min_val
                ).astype("float32")
                test_y_norm = (
                    ((max_val - min_val) * (test_y - norm_params[group].min_y))
                    / (norm_params[group].max_y - norm_params[group].min_y)
                ) + min_val
            else:
                test_X_norm = (
                    (
                        ((max_val - min_val) * (test_X - norm_params["all"].min_x))
                        / (norm_params["all"].max_x - norm_params["all"].min_x)
                    )
                    + min_val
                ).astype("float32")
                test_y_norm = (
                    ((max_val - min_val) * (test_y - norm_params["all"].min_y))
                    / (norm_params["all"].max_y - norm_params["all"].min_y)
                ) + min_val
            if self.classification:
                test_y_norm = test_y_norm.astype("int32")
            else:
                test_y_norm = test_y_norm.astype("float32")

            if ~self.classification and self.resp_width != 0:
                test_y_norm = test_y_norm.reshape(
                    test_y_norm.shape[0], test_y_norm.shape[1] * self.response
                )

            train_test_dict = {"test_X": test_X_norm, "test_y": test_y_norm}

            return EasyDict(train_test_dict)


def make_start_end_index_dict(end_idx: int) -> dict:
    """
    Create a dictionary with the key as the group and the value a list with the start and
    end indices of that group in the final numpy array. The final time series array
    has all groups concatenated so this will allow for those groups to be pulled out
    so individual groups can be analyzed.

    Parameters:
    -------
        end_idx: int
            The ending index of each group.

    Returns:
    -------
    dict: group_indicies
    """
    group_indices = {}
    end_idx_accum = list(itertools.accumulate([x[1] + 1 for i, x in enumerate(end_idx)]))
    for i, (group, end_idx) in enumerate(zip([x[0] for x in end_idx], end_idx_accum)):
        if i == 0:
            group_indices[group] = (0, end_idx)
        else:
            group_indices[group] = [end_idx_accum[i-1], end_idx]
    return group_indices
