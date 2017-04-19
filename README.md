Predicting Fantasy Football Points Using Machine Learning
=========================================================
by Zefeng Zhang, Donny Chen, Eric Lehman, Philip Rotella

Introduction
------------

FanDuel Inc. is a daily fantasy company that allows for legal gambling on multiple sports on a daily basis.  For NFL games, FanDuel allows you to select a lineup given several constraints.  Typically, you will have a set salary cap and from that salary cap you can spend money on players to set a fantasy lineup.  The most common lineup format is as shown in **Table 1** below.

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

* Season-to-Date Features

Fifty different statistics for each player are included in the model.  The term “season-to-date” indicates that a rolling average for each statistic is calculated up to but not including the “current” week’s game.  Intuitively, one would expect that past good performance would dictate future good performance.  Examples of statistics are passing yards, passing touchdowns, interceptions thrown, rushing yards, receptions and fumbles lost.

* Game Characteristics

Binary indicator variables were created for several additional features.  For a given player for a given week, dummy variables were created to indicate whether the player is playing the game at home or away.  Intuitively, one would expect a player to perform better during home games.  Dummy variables were also created for the player’s team as certain teams are more offensive-oriented.  Finally, dummy variables for the player’s opponent are included in the model.  The quality of the team that a player is playing will influence the amount of points scored.

* Additional Features Considered but Not Included in Model

Several other variables were considered for the model, but not included due to lack of availability or impracticality of using the data.  Injury information on the opposing team and/or a given player’s team would provide valuable information in a given week.  If the star quarterback of a given player’s team is not playing, it is likely that the wide receivers on that team would have reduced fantasy points.  Conversely, if a star defensive player is missing on the opposing team, it is likely that an offensive player will have increased points for the week.  Additional variables considered were specific opposing team defensive statistics and prior year fantasy points / statistics.

One additional approach considered was to include various fantasy website fantasy projections as features in the model.  Websites such as espn.com, fantasydata.com, yahoo.com, nfl.com, etc. all produce weekly projections for each player likely based on methodologies similar to those shown in this report.  It was the author’s opinion that the most interesting immediate task would be to compare the RMSE of the models developed herein to other websites’ projections. However, since the end goal is to produce the best FanDuel lineup, using other projections as features would be an interesting approach.  This will be discussed further in the Future Work / Conclusion section of this report.