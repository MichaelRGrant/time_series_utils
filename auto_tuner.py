import re
from typing import Callable, Dict, Tuple

from easydict import EasyDict
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
from skopt import BayesSearchCV
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


class AutoTune:
    """
    Using Bayesian optimization from skopt, tunes the model
    then runs the top_n best tunes and validates them choosing
    the best overall model.

    Parameters:
    -------
    model: Callable
    data: dict
        A dictionary of the training, validation, and testing
        data. The data should be in the form of a numpy array
        (N, timesteps, features)
    param_dict: dict
        The search space of the params in the model
    scoring_func: str
        The function used to score the model. Currently
        only accepts `auroc`
    epochs: int
        The number of epochs to use when validating the best
        tunes. This number should be set high because early stopping
        callbacks are used.
    """

    def __init__(
        self,
        model: Callable,
        data: Dict[str, Dict[str, Dict]],
        param_dict: Dict,
        scoring_func: str = "roc_auc",
        epochs: int = 30,
    ):
        self.model = model
        self.data = data
        self.param_dict = param_dict
        self.epochs = epochs
        if scoring_func == "roc_auc":
            self.high_is_good = True
            self.scoring_func = metrics.make_scorer(metrics.roc_auc_score)
        else:
            raise ValueError("Only currently using AUROC as the scoring function.")

        self.es = callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=6,
            min_delta=0.0005,
            restore_best_weights=True,
        )

    def bayes_sweep(
        self,
        cv: int = 3,
        n_iter: int = 10,
        refit: bool = False,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        """
        Run the BayesSearchCV hyperparameter tuner.

        Parameters:
        -------
        cv: int
            The number of cross validation folds to use for tuning
        n_iter: int
            The number of parameter sets to try
        refit: bool
        n_jobs: int
        verbose: int
        """
        keras_model = KerasClassifier(clone(self.model))
        sweeper = BayesSearchCV(
            estimator=keras_model,
            search_spaces=self.param_dict,
            scoring=self.scoring_func,
            n_iter=n_iter,
            cv=cv,
            refit=refit,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        sweeper.fit(self.data.train.train_X, self.data.train.train_y)

        self.sweeper = sweeper

    def validate_best_tunes(self, n_jobs: int, top_n_tunes: int) -> pd.DataFrame:
        """
        Validate the top_n best tunes.

        Parameters:
        -------
        n_jobs: int
        top_n_tunes: int
            Choose the number of top tunes to validate using the
            validation set.

        Returns:
        -------
        Dataframe of the top_n validated scores.
        """
        # TODO figure out how to save the fit models to save the best one
        # TODO create a shared list that each job can save to and then save the best model from there.
        _, params = self.get_tune_results()
        scores = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.fit_and_score)(clone(self.model), params[tune_rank], tune_rank)
            for tune_rank in range(top_n_tunes)
        )
        scores_df = pd.concat(scores)
        self.valid_best_tunes = scores_df

        self.fit_best_model()

        return scores_df

    def fit_and_score(
        self, cloned_model: tf.keras.models.Model, param_set: Dict, tune_rank: int
    ) -> pd.DataFrame:
        """
        Fit and score each set of parameters from the tuning process.

        Parameters:
        -------
        cloned_model: tf.keras.models.Model
            The model must be deep copied to be run in parallel.
        param_set: dict
            A set of parameters to fit the model.
        tune_rank: int
            The rank of that parameter_set from the tuning process.

        Returns:
        -------
        A dataframe row with the results from a model fitted with the
        `param_set` parameters.
        """
        np.random.seed(84)
        tf.random.set_seed(84)
        validation_results_df = pd.DataFrame()

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=10,
            min_delta=0.0005,
            restore_best_weights=True,
        )

        batch_size = param_set.pop("batch_size")
        _ = param_set.pop("epochs")
        cloned_model = cloned_model(**param_set)

        history = cloned_model.fit(
            x=self.data.train.train_X,
            y=self.data.train.train_y,
            validation_split=0.1,
            epochs=self.epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[es],
            verbose=0,
        )

        model_metrics = self.get_metrics(cloned_model, threshold=0.5)
        model_metrics.update(
            {"epochs": len(history.epoch), "response": self.data.response}
        )
        validation_results_df = validation_results_df.append(
            pd.DataFrame(model_metrics, index=[tune_rank])
        )

        return validation_results_df

    def fit_best_model(self):
        """
        After validation the top parameter sets from the tune,
        run train the model for a final time and save the model
        and history into the class.
        """
        _, params = self.get_tune_results()
        best_idx = self.valid_best_tunes["auroc"].idxmax()
        best_param_set = params[best_idx]

        batch_size = best_param_set.pop("batch_size")
        _ = best_param_set.pop("epochs")
        epochs = self.valid_best_tunes.loc[0]["epochs"]

        history = self.model.fit(
            x=self.data.train.train_X,
            y=self.data.train.train_y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=0,
        )

        horizon = self.data.train.train_X.shape[1]
        response = self.data.response

        self.best_history = history
        self.model.save(
            f"./models/horizon-{horizon}day_{response}"
        )

    def get_metrics(
        self, cloned_model: tf.keras.models.Model, threshold: float = 0.50
    ) -> Dict[str, float]:
        """
        Collect model metrics and output to a dictionary.

        Parameters:
        -------
        cloned_model: tf.keras.models.Model
        threshold: float
            The probability threshold to create a 1 or 0 prediction.

        Returns:
        -------
        Dictionary with all model metrics calculated.
        """

        yhats = cloned_model.predict(
            self.data.validation.test_X,
        )
        ypreds = np.where(yhats.ravel() > threshold, 1, 0)

        results_dict = {}

        auroc = metrics.roc_auc_score(self.data.validation.test_y, yhats)
        brier = metrics.brier_score_loss(self.data.validation.test_y, yhats)
        kappa = metrics.cohen_kappa_score(self.data.validation.test_y, ypreds)
        cr = metrics.classification_report(
            self.data.validation.test_y, ypreds, output_dict=True
        )
        specificity = cr["0"]["recall"]
        sensitivity = cr["1"]["recall"]

        results_dict = EasyDict(
            {
                "auroc": auroc,
                "brier": brier,
                "kappa": kappa,
                "specificity": specificity,
                "sensitivity": sensitivity,
                "threshold": threshold,
            }
        )

        return results_dict

    def get_tune_results(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract the parameters and metrics from the BayesSearchCV output.

        Returns:
        -------
        A tuple with the tune results as a dataframe and a dictionary of the
        parameters from each tune used to fit subsequent models.
        """
        score_df = pd.DataFrame(self.sweeper.cv_results_)
        drop_cols = [
            "params",
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "mean_test_score",
            "std_test_score",
        ]

        score_df.insert(0, "score", score_df["mean_test_score"])
        score_df.insert(1, "std", score_df["std_test_score"])
        score_df = (
            score_df.drop(columns=drop_cols)
            .sort_values("rank_test_score", ascending=self.high_is_good)
            .drop(columns=["rank_test_score"])
            .reset_index(drop=True)
        )
        params = {}
        params_df = score_df[
            [col for col in score_df.columns if re.match(r"^param", col)]
        ]
        params_df.columns = [re.sub(r"param_", "", col) for col in params_df.columns]
        for idx, row in params_df.iterrows():
            params[idx] = row.to_dict()

        return (
            score_df,
            params,
        )
