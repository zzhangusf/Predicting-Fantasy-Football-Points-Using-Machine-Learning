Predicting Fantasy Football Points Using Machine Learning
=========================================================
by Zefeng Zhang, Donny Chen, Eric Lehman, Philip Rotella

Introduction
------------

FanDuel Inc. is a daily fantasy company that allows for legal gambling on multiple sports on a daily basis.  For NFL games, FanDuel allows you to select a lineup given several constraints.  Typically, you will have a set salary cap and from that salary cap you can spend money on players to set a fantasy lineup.  The most common lineup format is as shown in **Table 1** below.

Table 1: Common Lineup Format for FanDuel Site

| POSITION      | NUMBER OF PLAYERS   |
| ------------- |:-------------:| 
| Quarterback   | 1      | 
| Running Back  | 2      |  
| Wide Receiver | 3      | 
| Tight End     | 1      | 
| Kicker        | 1      | 
| Defense       | 1      | 

Once lineups have been set, fantasy teams gain points via actual NFL football game statistics.  For example, typically a running back will receive 1 point for every ten yards rushing in a game.  Different leagues have different point settings.  For the analysis conducted in this report we assume standard PPR league scoring.  In a PPR league a player is awarded 0.5 points for every reception.

Data
----

Data from multiple sources were used.  The following R package was used to scrape data from the NFL’s API website:  https://github.com/maksimhorowitz/nflscrapR.  NFL player stats are available for all games from 2009 through 2016.  For the models created in this report, 50 different statistics were used.  Example R scripts have been uploaded on Canvas.  Player data has been scraped for the 2015 season and for weeks 1-12 of the 2016 season.

Additionally, FanDuel player salaries and point totals were uploaded from http://rotoguru.com.  The player’s point total will be the response variable that the machine learning algorithms will be trying to predict for each player.

 The final data source is projected fantasy player data for the top 50 players at each position (excluding team defenses).  This data was available from http://fantasydata.com. A python script was written to join the data for all players for all weeks in 2015 and 2016.  The appropriate python scripts have been uploaded to Canvas.

Joining the data was not a completely straightforward process as each data set had different player identification numbers, some players have similar names, and team names were not always abbreviated consistently.  Several verification steps were taken to make sure that the data was joined in a consistent manner.

Feature Selection
-----------------

Fifty different statistics for each player are included in the model.  The term “season-to-date” indicates that a rolling average for each statistic is calculated up to but not including the “current” week’s game.  Intuitively, one would expect that past good performance would dictate future good performance.  Examples of statistics are passing yards, passing touchdowns, interceptions thrown, rushing yards, receptions and fumbles lost.

* **Season-to-Date Features**

Fifty different statistics for each player are included in the model.  The term “season-to-date” indicates that a rolling average for each statistic is calculated up to but not including the “current” week’s game.  Intuitively, one would expect that past good performance would dictate future good performance.  Examples of statistics are passing yards, passing touchdowns, interceptions thrown, rushing yards, receptions and fumbles lost.

* **Game Characteristics**

Binary indicator variables were created for several additional features.  For a given player for a given week, dummy variables were created to indicate whether the player is playing the game at home or away.  Intuitively, one would expect a player to perform better during home games.  Dummy variables were also created for the player’s team as certain teams are more offensive-oriented.  Finally, dummy variables for the player’s opponent are included in the model.  The quality of the team that a player is playing will influence the amount of points scored.

* **Additional Features Considered but Not Included in Model**

Several other variables were considered for the model, but not included due to lack of availability or impracticality of using the data.  Injury information on the opposing team and/or a given player’s team would provide valuable information in a given week.  If the star quarterback of a given player’s team is not playing, it is likely that the wide receivers on that team would have reduced fantasy points.  Conversely, if a star defensive player is missing on the opposing team, it is likely that an offensive player will have increased points for the week.  Additional variables considered were specific opposing team defensive statistics and prior year fantasy points / statistics.

One additional approach considered was to include various fantasy website fantasy projections as features in the model.  Websites such as espn.com, fantasydata.com, yahoo.com, nfl.com, etc. all produce weekly projections for each player likely based on methodologies similar to those shown in this report.  It was the author’s opinion that the most interesting immediate task would be to compare the RMSE of the models developed herein to other websites’ projections. However, since the end goal is to produce the best FanDuel lineup, using other projections as features would be an interesting approach.  This will be discussed further in the Future Work / Conclusion section of this report.

Models
------

Five separate machine learning algorithms were used to predict player fantasy points: ridge regression, bayesian ridge regression, elastic net regularization, random forest and gradient boosting.  For each algorithm, separate models were developed for each player position: quarterback, running back, wide receiver, tight end and place kicker.  The data were grouped randomly into training sets (70% of data) and test sets (30% of the data) in order to predict the RMSE (root mean squared error) for each model at each position.  Ultimately, the models were used to predict the RMSE for the top 50 fantasy players at each position (per http://fantasydata.com) in both 2015 and 2016 (weeks 5 through 12).  The comparisons to the 2015 data result in either a training RMSE or cross validation RMSE.  For testing, we compare to week 5 through 12 data in 2016.  The thinking is that the models will work best with more “rolling average” data, hence from a predictor’s standpoint (read: a bettor’s standpoint), predicting from weeks 5 forward provide the best chance to correctly predict fantasy points.  The RMSEs obtained were compared to the FanDuel predictions made by http://fantasydata.com.

All models were implemented in Python using the scikit-learn package.  During the training/cross validation phase, parameters were tuned in an attempt to find the lowest RMSE.

* **Ridge Regression**

Ridge regression is similar to linear regression however it contains a penalty term which increases as the feature coefficients increase. An alpha of 1 was used.

* **Bayesian Ridge Regression**

Bayesian ridge regression is similar to ridge regression however it includes information about the features to determine the penalty weight.  The models used the scikit-learn default fitting parameters and used the “Compute Score = True” option which re-computes the objective function at each step of the model.

* **Elastic Net Regularization**

Elastic net regularization applies a weighted average of the ridge regression and lasso regression penalties.  Alpha = 1.0 and a mixing parameter of 0.5 were used for the model.

* **Random Forest**

Random forest is a tree-based machine learning algorithm which splits on randomly generated selection features in an attempt to prevent over-fitting.  The number of estimators used was 10 with a minimum sample split of 2.

* **Gradient Boosting**

Gradient Boosting is also a tree-based method which learns from previous performance mistakes.  A grid search was performed to optimize the parameters within the model.  The optimal model parameters were a learning rate of 0.3 and number of estimators = 50.

Results
-------

* **Cross validation**

K-folds (k = 5) cross validation was performed using random samples of 30% of the data as the test data.  This was performed for models for all 5 player positions.  Different methods produced the best results for different positions.  Figure 1 shows the average RMSEs for each model during the train, cross validation phases for wide receivers.  Information about the testing phase is discussed in the next section.

Figure 1: RMSE Comparison among Different ML Models

![alt text](https://github.com/zefeng-zhang/Predicting-Fantasy-Football-Points-Using-Machine-Learning/blob/master/images/rmse_model.png)

* **Testing: comparisons to 2016 FantasyData.com Predictions**

In order to test the models, the trained models described in the previous section were tested against actual FanDuel points for weeks 5 through 12 in the 2016 football season.  The resulting RMSEs were compared to the RMSEs obtained by using FantasyData.com predictions for the same weeks.  It is seen in Figure 2 that the RMSEs are on the same order of magnitude as the FantasyData.com predictions.  It should be noted that analysts are employed by various websites to produce fantasy football predictions who likely have more time and resource to develop robust prediction models.  The fact that the RMSEs are very close is a good sign.  Ultimately, with the addition of some of the variables described in the Feature Selection section of this report, it is the opinion of the authors that the RMSE would likely outperform the FantasyData.com projections.

Please note that the results shown in **Figure 2** are for for the top performing models for each position, which generally were some form of ridge regression.  Ridge regression in some sense acts as a shrinkage method as it attempts to reduce the coefficients of unnecessary variables. 

Figure 2: Test RMSE (Comparison to FantasyData.com Predictions, 2016 Weeks 5-12)

![alt text](https://github.com/zefeng-zhang/Predicting-Fantasy-Football-Points-Using-Machine-Learning/blob/master/images/rmse_position.png)

Conclusion and future work
--------------------------

Machine learning models predicting fantasy football points were successfully implemented using ridge regression, bayesian ridge regression, elastic net, random forest and boosting.  The models performed reasonably well when compared to FantasyData.com projections for the 2016 season.  In addition lineup optimizations were performed for the 2016 season.

If the project were to be continued several additional steps could be taken to potentially improve model performance:

* Add additional variables into the model such as injuries, defensive stats and/or previous year fantasy performance.
* Add additional fantasy projections into the model
* Try additional modelling techniques and parameter optimizations
* Algorithmically perform feature selection (possibly could drop some of the features used)





