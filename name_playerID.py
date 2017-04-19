import pandas as pd

data2015 = pd.read_csv('aggregated_2015.csv')

data2015['identity'] = data2015['name'] + data2015['Team'] + data2015['Pos']

old_identity = None
old_playerID = None
# for i in range(2):
for i in range(len(data2015.index)):
#     print data2015.iloc[i,:]['identity']
    new_identity = data2015.iloc[i,:]['identity']
    new_playerID = data2015.iloc[i,:]['playerID']
    if(old_identity == new_identity and old_playerID != new_playerID):
        print i
        print old_playerID, old_identity
        print new_playerID, new_identity
    old_identity = new_identity
    old_playerID = new_playerID



