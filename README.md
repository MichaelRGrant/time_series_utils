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
Also since the output for the training and testing data is a numpy array, the dictionary has a key `group_idx` which
is a dictionary of group names for the keys and start and end indicies for each group. This allows the user to easily know 
how the model is performing on each individual group. This is useful during model evaluation. 

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
which allows for direct calling of dicionary keys like `train.train_X`, or `test.test_X` for the training and testing
sets, respectively. 
