# Time Series Utilities

### Modules
**window.py**

This module takes in data and returns a 3D windowed dataframe of the form (N, timesteps, features) 
used for deep learning time series analysis. The script will take data, normalize either using a minmax
or center-scale routine, and also split the data. Normalization parameters are saved from the training set
and used to normalize the validation/testing sets as well. 

This module has the flexibility to either predict some future timepoint, the `horizon` but can also predict the current time
point. In addition, if the user wants to predict at many timepoints in the future for example from days 1:10 this 
can also occur.

Another feature of this module is the ability to normalize over the entire data or to normalize by individual groups. 
Also since the output for the training and testing data is a numpy array, the EasyDict5 output of the 
`make` method has a key `group_idx` which is a dictionary of group names for the keys and start and end 
indicies for each group. This allows the user to easily know how the model is performing on each individual group. 
This is useful during model evaluation. 

```
window = Window(
    seq_length=sequence_length,
    horizon=horizon,
    feat_cols=feat_cols,
    resp_cols=resp_cols,
    group_col="symbol",
    resp_width=resp_width,
    classification=True,
    predict_current=True,
    scale_by_group=True, 
)

train = window.make(
    dataset=model_train_df,
    split=None,
    scale="minmax",
    test_set=False,
)

test = window.make(
    dataset=model_val_df,
    split=None,
    scale="minmax",
    test_set=True,
    norm_params=window.norm_params,
)
```

The above code makes two datasets for training and testing. The output of the `make` method is an `EasyDict` 
which allows for direct calling of dictionary keys in the form like `train.train_X`, or `test.test_X` for the training and testing
sets, respectively.
---
**auto_tuner.py**

The AutoTune class in this module uses `skopt`'s `BayesSearchCV` functionality to tune a time-series 
deep learning model. The model and search space must be defined by the user, and the class does the rest.
A tune is performed, tested via cross validation, and then some number of the best tunes is validated
using validation data. The best model is then fit and saved. 

After this class is run, a dataframe with the best tunes, the parameters for those tunes,
a dataframe of model metrics with of the validated best tunes, and the history object of the best
model fit are all saved to the class; this can then be pickled. 

```
bayes_model_tuner = AutoTune(keras_model, train_test_data, bayes_param_grid, epochs=25)
bayes_model_tuner.bayes_sweep(n_iter=10)
scores_df = bayes_model_tuner.validate_best_tunes(
    n_jobs=multiprocessing.cpu_count(), top_n_tunes=5
)
```

The above function requires that the model created be wrapped in the `KerasClassifier` class
to make it an sklearn compatiable model. This is needed for the parameter sweeper, `BayesSearchCV`.
Also the dictionary `train_test_data` is an EasyDict, to make direct calling of keys possible, i.e.
`dict.key` instead of `dict["key"]` and should be of the form:
```
train_test_.data.train.train_X
train_test_data.train.train_y
train_test_data.validation.test_X
train_test_data.validation.test_y
train_test_data.test.test_X
train_Test_data.test_test_y
```