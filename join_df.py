from nameparser import HumanName
import pandas as pd

"""
Load data from multiple sources
The data include Statistics, Fanduel Points, and Fantasy Projections
"""

df_abbr = pd.read_csv("all_nfl_team_abbr.csv")
fanduel_2015 = df_abbr['fanduel_2015']
fanduel_2016 = df_abbr['fanduel_2016']
statistics_2015 = df_abbr['statistics_2015']
statistics_2016 = df_abbr['statistics_2016']
fantasy_2015 = df_abbr['fantasy_2015']
fantasy_2016 = df_abbr['fantasy_2016']

"""
Normalize team name and player name
"""

# df1: 2015_Fanduel_Results
# df2: players_box_scores_2015.csv
# df3: FD_2015_Projections.csv (data from Fantancydata.com)

def normalize_team_name(df1, abbr):
    """
    Team name: from full name to abbreviation
    """
    ref = statistics_2015
    subs = dict(zip(abbr, ref))
    df1['Team'] = [subs.get(item) for item in df1['Team']]
    try:
        df1['Oppt'] = [subs.get(item) for item in df1['Oppt']]
    except:
        pass

def normalize_fanduel_player_name(df1, special_names):
    """
    Player name: from full name to abbreviation
    """
    names = []
    for i in range(len(df1.index)):
        name = HumanName(df1.ix[i, 'name'])
        initial = name.first[0]
        new_name = "%s.%s" % (initial, name.last)

        # Determine if we should use two letters to identify this player
        temp = new_name + df1.ix[i, 'Team']
        if(temp in special_names):
            new_name = "%s.%s" % (name.first[0:2], name.last) 

        names.append(new_name)
    df1['name'] = names

def normalize_statistics_player_name(df2):
    """
    Find special names that require two letters to identify this name
    """
    special_names = []
    name_team = df2['name'] + df2['Team']
    for i in range(len(name_team.index)):
        name = name_team.iloc[i]
        if name[1] != '.':
            begin = name[:1]
            end = name[2:]
            name = begin + end
            special_names.append(name)
    return special_names

def normalize_projection_player_name(df3, special_names):
    """
    Player name: from full name to abbreviation
    """
    names = []
    for i in range(len(df3.index)):
        name = df3.ix[i, 'name']
        name = name.split(' ')
        new_name = "%s.%s" % (name[0][0], name[-1])

        # Determine if we should use two letters to identify this player
        temp = new_name + df3.ix[i, 'Team']
        if (temp in special_names):
            new_name = "%s.%s" % (name[0][0:2], name[-1]) 

        names.append(new_name)
    df3['name'] = names

"""
Main program:
Aggregate datasets for both 2015 and 2016 Seasons
"""

years = ['2016', '2015']
for year in years:

    df1 = pd.read_csv(year + "_Fanduel_Results",delimiter=';')
    df1 = df1.rename(columns = {'Week':'weeks','Name':'name', 'Year':'Season'})

    df2 = pd.read_csv("players_box_scores_" + year + ".csv")
    df2.ix[df2['Team']=='JAX','Team'] = 'JAC'

    df3 = pd.read_csv("FD_" + year + "_Projections.csv")
    df3 = df3.rename(columns = {'week':'weeks','player':'name', 'team':'Team'})
    df3.ix[df3['Team']=='JAX','Team'] = 'JAC'

    if(year == '2015'):
        normalize_team_name(df2, statistics_2015)
    else:
        normalize_team_name(df2, statistics_2016)
    special_names = normalize_statistics_player_name(df2)

    if(year == '2015'):
        normalize_team_name(df1, fanduel_2015)
    else:
        normalize_team_name(df1, fanduel_2016)
    normalize_fanduel_player_name(df1, special_names)

    if(year == '2015'):
        normalize_team_name(df3, fantasy_2015)
    else:
        normalize_team_name(df3, fantasy_2016)
    normalize_projection_player_name(df3, special_names)

    print(sorted(df1['Team'].unique()))
    print(sorted(df2['Team'].unique()))
    print(sorted(df3['Team'].unique()))

    df = df1.merge(df2, how='inner', on=['weeks','Team','name']).\
            merge(df3, how='inner', on=['weeks','Team','name'])
    df.to_csv("aggregated_"+year+".csv", index=None)
