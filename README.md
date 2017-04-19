Predicting Fantasy Football Points Using Machine Learning
=========================================================
by Zefeng Zhang, Donny Chen, Eric Lehman, Philip Rotella

Introduction
------------

Fantasy football is a multibillion dollar industry and has seen a global rise in popularity over the last decade.  Weekly and daily fantasy leagues are available across multiple platforms and can be played for monetary incentive or just for fun.  The ability to correctly predict points for National Football League (NFL) players will lead to victories in fantasy football matchups, provide bragging rights, and can also lead to financial earnings if playing in a “for-money” league.

This report presents results of machine learning techniques used to attempt to predict fantasy points for NFL players on a weekly basis.  The report will present information about fantasy football, data sources, machine learning techniques and results for each technique.  This report also references additional Python and R scripts that will be uploaded to the Canvas website.

Background
----------

FanDuel Inc. is a daily fantasy company that allows for legal gambling on multiple sports on a daily basis.  For NFL games, FanDuel allows you to select a lineup given several constraints.  Typically, you will have a set salary cap and from that salary cap you can spend money on players to set a fantasy lineup.  The most common lineup format is as shown in Table 1 below.

| POSITION      | NUMBER OF PLAYERS   |
| ------------- |:-------------:| 
| Quarterback   | 1      | 
| Running Back  | 2      |  
| Wide Receiver | 3      | 
| Tight End     | 1      | 
| Kicker        | 1      | 
| Defense       | 1      | 
