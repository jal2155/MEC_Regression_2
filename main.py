import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('caiso_data.csv', parse_dates=[0], index_col=0)
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.weekday
test_param = 'Loss'

print(df[['SP15', 'Loss', 'Congestion']].corr(method='kendall'))


def gradient_boosting():
    params_file_name = 'archive/best_params_loss_year.csv'
    X, y = df[['SP15', 'Month', 'DayOfWeek']], df[test_param]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    GBR = GradientBoostingRegressor()

    if path.exists('best_params_%s.csv' % test_param.lower()):
        best_params = pd.read_csv(params_file_name, index_col=0).squeeze().to_dict()
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_samples_split'] = int(best_params['min_samples_split'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        print(best_params)
    else:
        param_grid = {'n_estimators': [(x + 2) * 100 for x in range(5)],
                      'max_depth': [2, 3, 4],
                      'min_samples_split': [5, 6, 7],
                      'learning_rate': [round(0.01 * (x + 1), 2) for x in range(20)],
                      }

        clf = GridSearchCV(estimator=GBR, param_grid=param_grid, n_jobs=-1)
        clf.fit(X_train, y_train)

        print('Best parameter set found on development')
        print(clf.best_params_)
        best_params = clf.best_params_
        pd.DataFrame(data=best_params, index=[0]).to_csv('best_params_%s.csv' % test_param.lower())

    reg = GradientBoostingRegressor(**best_params)
    reg.fit(X_train, y_train)
    temp = pd.Series(reg.predict(X_test))
    temp.index = X_test.index
    loss_forecast = pd.concat(objs=[y_test, temp], axis=1)
    loss_forecast.columns = ['Observed', 'Predicted']
    loss_forecast.sort_index().to_csv('predicted_%s.csv' % test_param.lower())

    plt.clf()
    sns.set_style('whitegrid')
    sns.lineplot(data=loss_forecast, dashes=False)
    plt.title('Historical vs. Predicted %s @ SP-15' % test_param.lower())
    plt.tight_layout()
    plt.savefig('predicted_%s.png' % test_param)

    mse = np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))
    print('The mean squared error (MSE) on test set is {:.2f}'.format(mse))
    mae = mean_absolute_error(y_test, reg.predict(X_test))
    print('The mean absolute error (MAE) on test set is {:.2f}'.format(mae))
    avg_loss = df[test_param].abs().mean()
    print(avg_loss)


def rolling_stuff():
    X = df[['SP15', 'Congestion', 'Loss']]
    y = df['MEC']
    dic = {}
    for x in X:
        reg = linear_model.LinearRegression()
        temp_x = df[x].values.reshape(-1, 1)
        fit = reg.fit(temp_x, y.values.reshape(-1, 1))
        params = reg.coef_
        score = reg.score(temp_x, y)
        print(params)
        print(score)

    lag = 365
    x_name = 'Loss'
    roll_mu = df[x_name].rolling(window=lag).mean()
    roll_sd = df[x_name].rolling(window=lag).std()
    roll_sk = df[x_name].rolling(window=lag).skew()
    roll_kt = df[x_name].rolling(window=lag).kurt()

    df2 = pd.concat(objs=[roll_mu, roll_sd, roll_sk, roll_kt], axis=1).dropna()
    df2.columns = ['Mean', 'StDev', 'Skew', 'Kurtosis']
    print(df2)

    # Reject null hypothesis (p < 0.05) --> data are stationary
    adf_result = adfuller(df2['Mean'])
    print('p-value: %s' % adf_result[1])
    print(adf_result)

    plt.clf()
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('Evolution of SP-15 %s Over Time' % x_name)

    sns.lineplot(ax=axes[0, 0], data=df2['Mean'], dashes=False)
    sns.lineplot(ax=axes[0, 1], data=df2['StDev'], dashes=False)
    sns.lineplot(ax=axes[1, 0], data=df2['Skew'], dashes=False)
    sns.lineplot(ax=axes[1, 1], data=df2['Kurtosis'], dashes=False)

    plt.tight_layout()
    plt.savefig('sp_15_losses.png')


gradient_boosting()
