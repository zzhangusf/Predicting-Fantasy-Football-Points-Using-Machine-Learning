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
    df = pd.read_csv(file_name)
    df.sort_values(by = ['playerID', 'weeks']) # for calculating rolling average
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
    #Game Characteristic Indicators (home/away, opponent, team)
    df, game_features = get_game_char_indicators(df)
    #Player Statistic Features (Season, last 4 weeks, previous week)
    df, player_features = get_player_averages(df, stats)
    #Combine features and return complete df and feature names
    features = game_features + player_features
    df = df.fillna(0)
    return df, features

def get_game_char_indicators(df):
    """
    Adds game indicator variables returns column names
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
    #return list of feature column names
    return df, ['home'] + list(oppts.columns) + list(team_list)

def rolling_average(df, window):
    return df.rolling(min_periods=1, window=window).mean().shift(1)

def get_player_averages(df, stats):
    """
    Adds player averages for all stats and FanDuel point histories,
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

try:
    train = pd.read_pickle('train_df.p')
    features = pickle.load(open('train_features.p', 'rb'))
except:
    train, features = data_processing('aggregated_2015.csv')
    pickle.dump(train, open('train_df.p', 'wb'))
    pickle.dump(features, open('train_features.p', 'wb'))

try:
    test = pd.read_pickle('test_df.p')
    features2 = pickle.load(open('test_features.p', 'rb'))
except:
    test, features2 = data_processing('aggregated_2016.csv')
    pickle.dump(test, open('test_df.p', 'wb'))
    pickle.dump(features2, open('test_features.p', 'wb'))

train, features = data_processing('aggregated_2015.csv')
pickle.dump(train, open('train_df.p', 'wb'))
pickle.dump(features, open('train_features.p', 'wb'))
test, features2 = data_processing('aggregated_2016.csv')
pickle.dump(test, open('test_df.p', 'wb'))
pickle.dump(features2, open('test_features.p', 'wb'))

if (features != features2):
    print "Debug error about feature inconsistency"
    exit()
response = 'FD points'

# Dataframe initialization
positions = sorted(train['Pos'].unique())
estimators = ["Ridge",
              "ElasticNet",
              "RandomForestRegressor"
              # "GradientBoostingRegressor"
              # "SVM"
              ]
rmse_types = ['train', 'cv', 'test']
rmse_names = [x + '_' + y for y in rmse_types for x in estimators]
df_rmse = pd.DataFrame([[0.0 for i in range(len(positions))] for j in range(len(rmse_names))], 
    index = rmse_names, columns = positions)

# Iterate through all positions
for position in positions:
    print ('Learning for Position %s ...' % position)
    df_pos_train = train.ix[train['Pos'] == position,]
    df_pos_test = test.ix[test['Pos'] == position,]

    for i in range(len(estimators)):
        est = estimators[i]

        if(est == "GradientBoostingRegressor"): #Gradient Boosting Regressor with Grid Search for hyperparameters
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

        # Fit testing results using the fitted model above
        train_rmse = np.sqrt(np.mean( (df_pos_train['FD points'] - grid_search.predict(df_pos_train[features]))**2.0 ))
        test_rmse = np.sqrt(np.mean((df_pos_test['FD points'] - grid_search.predict(df_pos_test[features]))**2.0 ))
        # Deprecating "mean_squared_error". Use "neg_mean_squared_error" instead.
        cv_rmse = np.sqrt(np.abs( cross_val_score(grid_search, train[features], train['FD points'],\
            cv = 5, scoring = 'neg_mean_squared_error').mean() ))
        # Import values into dataframe
        for score_type in rmse_types:
            df_rmse.loc[estimators[i] + "_" + score_type, position] = eval(score_type + '_rmse')

        pickle.dump(grid_search, open(estimators[i] + "_" + position, 'wb'))

df_rmse.to_csv('rmse.csv', header = True, index=True)


"""
MSE of FD_2016_Projections.csv (Fantasydata.com)
"""

test['diff'] = (test['proj'] - test['FD points']) ** 2.0
FantasyData_rmse = (test.groupby(['Pos'])['diff'].mean()) ** 0.5
FantasyData_rmse.to_csv('FantasyData_rmse.csv', header = True, index = True)

print "Program finished normally"