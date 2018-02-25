import json_tricks
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasRegressor
from math import sqrt
from matplotlib import pyplot
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf


class Evaluate:
    def __init__(self, model, train_X, train_y, test_X, test_y, mean_y, std_y, seq_length, feat_cols, resp_cols):
        self.model = model
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.mean_y = mean_y
        self.std_y = std_y
        self.seq_length = seq_length
        self.feat_cols = feat_cols
        self.resp_cols = resp_cols
        self.dense = len(resp_cols)

    def compute_mae(self, model, X, y, reshape=False):
        forecasts = model.predict(X)
        if reshape:
            dense = forecasts.shape[1]
            forecasts = forecasts.reshape(forecasts.shape[0], dense)
        else:
            pass
        forecasts = (forecasts * self.std_y) + self.mean_y
        y = (y * self.std_y) + self.mean_y
        result = mean_absolute_error(y_pred=forecasts, y_true=y)
        return result

    def test_mae(self, model):
        return (self.compute_mae(model, X=self.train_X, y=self.train_y),
                self.compute_mae(model, X=self.test_X, y=self.test_y))

    def report(self, train_list, test_list):
        train_mean = np.asarray(train_list).mean()
        train_std = np.asarray(train_list).std()
        test_mean = np.asarray(test_list).mean()
        test_std = np.asarray(test_list).std()
        train = str('Training MAE %f ± %f'% (train_mean, train_std))
        test = str('Testing MAE %f ± %f'% (test_mean, test_std))
        return train, test

    def run_single_model(self, iter, verbose, best_params):
        np.random.seed(84)
        tf.set_random_seed(84)

        ITERATIONS = iter
        VERBOSE = verbose

        train_results, test_results = [], []

        print('Predicting current time, with a {0:.1f} hour window. '.format(self.seq_length))
        print('Feature space: ', self.feat_cols)
        print('Response: ', self.resp_cols)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-08, mode='auto', verbose=1)

        for i in range(ITERATIONS):
            model = KerasRegressor(build_fn=self.model, shuffle=True, verbose=VERBOSE, **best_params)
            fitted_model = model.fit(self.train_X, self.train_y, verbose=VERBOSE, validation_split=0.15,
                                     callbacks=[reduce_lr])
            train_results.append(self.test_mae(model)[0])
            test_results.append(self.test_mae(model)[1])
            print(self.report(train_results, test_results))
        print(self.report(train_results, test_results))
        return fitted_model, train_results, test_results, best_params

    def predict_error(self, fitted_model, train_results, test_results):
        yhat = fitted_model.predict(self.test_X, batch_size=len(self.test_X))
        yhat = yhat.reshape(yhat.shape[0], self.dense)
        yhat_dict = {}
        y_dict = {}
        resultsMAE = []
        resultsRMSE = []
        for i in range(yhat.shape[1]):
            if i == 0:
                name = 'leaf_wet'
            else:
                name = 'solar'

            yhat_dict[name] = (yhat[:, i] * self.std_y[i]) + self.mean_y[i]
            if name == 'leaf_wet':
                yhat_dict[name][yhat_dict[name] < 0] = 0  # coerce leaf wetness below 0 -> 0
                yhat_dict[name][yhat_dict[name] > 1] = 1
            elif name == 'solar':
                yhat_dict[name][yhat_dict[name] < 0] = 0  # coerce solar radiation below 0 -> 0
                yhat_dict[name][yhat_dict[name] > 1500] = 1500
            else:
                pass
            y_dict[name] = (self.test_y[:, i] * self.std_y[i]) + self.mean_y[i]
            name_rmse = sqrt(mean_squared_error(y_dict[name], yhat_dict[name]))
            name_mae = mean_absolute_error(y_pred=yhat_dict[name], y_true=y_dict[name])
            resultsMAE.append(str('Test MAE for %s: %.3f' % (name, name_mae)))
            resultsRMSE.append(str('Test RMSE for %s: %.3f' % (name, name_rmse)))

        results = self.report(train_results, test_results)
        results = str(results)

        return results, resultsMAE, resultsRMSE, yhat_dict, y_dict

    def save_results(self, path, results, resultsMAE, resultsRMSE, best_params):
        f = open(path, 'w')
        f.write('RESULTS:\n' + results + '\n')
        for i in range(len(resultsMAE)):
            f.write(resultsMAE[i] + '\n' + resultsRMSE[i] + '\n')
        f.write('\nPARAMETERS: \n' + json_tricks.dumps(best_params) + '\n \n')
        f.write('FEATURES: \n')
        for cols in self.feat_cols:
            f.write(str(cols) + '\n')
        f.write('\nRESPONSE: \n' + str(self.resp_cols))
        f.close()

    def plot_results(self, yhat_dict, y_dict, start, end):
        font = dict(family='sans-serif',
                    color='black',
                    weight='normal',
                    size=25)
        START = start
        END = end
        pyplot.style.use('ggplot')
        pyplot.figure(figsize=(16, 10))
        j = 1
        for i in range(len(yhat_dict)):
            pyplot.subplot(len(yhat_dict), 1, j)
            pyplot.plot(list(yhat_dict.items())[i][1][START:END], label='prediction', linewidth=2)
            pyplot.plot(list(y_dict.items())[i][1][START:END], label='true', linewidth=2, alpha=1)
            pyplot.ylabel(str(list(yhat_dict.keys())[i]), fontdict=font)

            pyplot.legend(loc=1, prop={'size': 20})
            pyplot.tick_params(labelsize=15)
            pyplot.xlabel('Hourly', fontdict=font)
            j = j + 1
            name = str(list(yhat_dict.keys())[i])
            path = '../figures/' + name + 'SEEED-DP-RH-WindSpeed-Gust_W48_H1_ALLDATA.png'
            print(path)
        # pyplot.savefig(save_path)
        pyplot.show()

