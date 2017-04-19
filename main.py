import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

try:
    import cpickle as pickle
except:
    import pickle

"""
Featuer Engineering
"""

def data_processing(file_name):
    """
    Read the csv files and create game characterstic features and player statistic features
    """
    df = pd.read_csv(file_name)
    df.sort_values(by = ['playerID', 'weeks']) # for rolling average
    # All box score player stats, except defensive statistics
    stats = ['pass.att', 'pass.comp', 'passyds', 'pass.tds', 'pass.ints',
             'pass.twopta', 'pass.twoptm', 'rush.att', 'rushyds', 'rushtds',
             'rushlng', 'rushlngtd', 'rush.twopta', 'rush.twoptm', 'recept',
             'recyds', 'rec.tds', 'reclng', 'reclngtd', 'rec.twopta',
             'rec.twoptm', 'kick.rets', 'kickret.avg', 'kickret.tds',
             'kick.ret.lng', 'kickret.lngtd', 'punt.rets', 'puntret.avg',
             'puntret.tds', 'puntret.lng', 'puntret.lngtd', 'fgm', 'fga',
             'fgyds', 'totpts.fg', 'xpmade','xpmissed','xpa','xpb','xppts.tot',
             'totalfumbs', 'fumbyds','fumbslost']
    # Game Characteristic Indicators, e.g. home/away, opponent, team
    df, game_features = get_game_char_indicators(df)
    # Player Statistic Features, e.g. Season, last 4 weeks, previous week
    df, player_features = get_player_averages(df, stats)
    features = game_features + player_features
    df = df.fillna(0)
    return df, features

def get_game_char_indicators(df):
    """
    Transform str cols into game categorical variables
    Returns transformed and columns
    """
    df['home'] = 1 * df['h/a'] == 'h'
    oppts = pd.get_dummies(df['Oppt'], prefix='Oppt')
    teams = pd.DataFrame()
    team_list = pd.Series(['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET',\
                           'GB', 'HOU', 'IND', 'JAC', 'KC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ',\
                           'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS'])
    for team in df['Team']:
        temp = (team_list == team)
        teams = teams.append(temp, ignore_index=True)
    teams.index = range(len(df['Team']))
    teams.columns = list(team_list)
    df = pd.concat([df, oppts, teams], axis=1)
    return df, ['home'] + list(oppts.columns) + list(team_list)

def rolling_average(df, window):
    return df.rolling(min_periods=1, window=window).mean().shift(1)

def get_player_averages(df, stats):
    """
    Estimate player averages for all stats and FanDuel point histories,
    for season-to-date, last 4 weeeks, and previous week
    """
    feature_names = []
    for stat in df[stats + ['FD points']]:
        df['season_{}'.format(stat)] = df.groupby('playerID')[stat].apply(lambda x: rolling_average(x, 16))
        df['recent_{}'.format(stat)] = df.groupby('playerID')[stat].apply(lambda x: rolling_average(x, 4))
        df['prev_{}'.format(stat)] = df.groupby('playerID')[stat].apply(lambda x: rolling_average(x, 1))
        feature_names = feature_names + [time + "_" + stat for time in ['season', 'recent', 'prev']]
    return df, feature_names

"""
Main Program
"""

# Read csv files
train, features = data_processing('aggregated_2015.csv')
test, features2 = data_processing('aggregated_2016.csv')
if (features != features2):
    print "Debug error about feature inconsistency"
    exit()

""" RMSE dataframe initialization """

# Dataframe cols, e.g. PK
positions = sorted(train['Pos'].unique())
estimators = ["Ridge",
              "ElasticNet",
              "RandomForestRegressor"
              # "GradientBoostingRegressor"
              # "SVM"
              ]
types = ['train', 'cv', 'test']
# Dataframe index, e.g. Ridge_train
rmse_names = [x + '_' + y for y in types for x in estimators]
# Initialize a matrix filled with 0s
df_rmse = pd.DataFrame([[0.0] * len(positions) for j in range(len(rmse_names))], 
    index = rmse_names, columns = positions)

""" Machine Learning """

for position in positions:
    # Iterate through all positions
    print ('Learning for Position %s ...' % position)
    df_pos_train = train.ix[train['Pos'] == position,]
    df_pos_test = test.ix[test['Pos'] == position,]

    for i in range(len(estimators)):
        est = estimators[i]

        if(est == "GradientBoostingRegressor"):
            n_estimators = [50]
            learning_rate = [0.1]
            param_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
            grid_search = GridSearchCV(GradientBoostingRegressor(max_depth=3), param_grid, cv=5)
            grid_search.fit(df_pos_train[features], df_pos_train['FD points'])

        elif(est == "RandomForestRegressor"):
            n_estimators = [50]
            param_grid = {'n_estimators': n_estimators}
            grid_search = GridSearchCV(RandomForestRegressor(max_depth=3), param_grid, cv=5)
            grid_search.fit(df_pos_train[features], df_pos_train['FD points'])

        elif(est == "ElasticNet"):
            grid_search = ElasticNetCV().fit(df_pos_train[features], df_pos_train['FD points'])

        elif(est == "BayesianRidge"):
            alpha_1 = [1e-6, 1e-5, 1e-7]
            alpha_2 = [1e-6, 1e-5, 1e-7]
            lambda_1 = [1e-6, 1e-5, 1e-7]
            lambda_2 = [1e-6, 1e-5, 1e-7]
            param_grid = {'alpha_1': alpha_1, 'alpha_2':alpha_2, 'lambda_1':lambda_1, 'lambda_2':lambda_2}
            grid_search = GridSearchCV(BayesianRidge(), param_grid, cv=5)
            grid_search.fit(df_pos_train[features], df_pos_train[target])

        elif(est == "Ridge"):
            grid_search = RidgeCV().fit(df_pos_train[features], df_pos_train['FD points'])

        elif(est == "SVM"):
            C = [50]
            gamma = [0.3]
            param_grid = {'C': C, 'gamma': gamma}
            grid_search = GridSearchCV(SVC(), param_grid, cv=5)
            grid_search.fit(df_pos_train[features], df_pos_train['FD points'])

        else:
            print est
            print "Cannot find the algorithm"
            exit()

        train_rmse = np.sqrt(np.mean( (df_pos_train['FD points'] - \
                    grid_search.predict(df_pos_train[features]))**2.0 ))
        test_rmse = np.sqrt(np.mean( (df_pos_test['FD points'] - \
                    grid_search.predict(df_pos_test[features]))**2.0 ))
        # Deprecating "mean_squared_error". Use "neg_mean_squared_error" instead.
        cv_rmse = np.sqrt(np.abs( cross_val_score(grid_search, train[features], train['FD points'],\
            cv = 5, scoring = 'neg_mean_squared_error').mean() ))

        # Given the variable name in a string, get the variable value and import into dataframe
        for val in types:
            df_rmse.loc[estimators[i] + "_" + val, position] = eval(val + '_rmse')

""" save rmse into csv """

df_rmse.to_csv('rmse.csv', header = True, index=True)

"""
MSE of FD_2016_Projections.csv (Fantasydata.com)
"""

test['diff'] = (test['proj'] - test['FD points']) ** 2.0
FantasyData_rmse = (test.groupby(['Pos'])['diff'].mean()) ** 0.5
FantasyData_rmse.to_csv('FantasyData_rmse.csv', header = True, index = True)

print "Program finished normally"