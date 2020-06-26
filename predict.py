# Getting required libraries
import os
import pandas as pd
import numpy as np
from joblib import load
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def read_and_process(id, date, team, venue):
    path = os.path.join('Excel Files','players','{}.csv'.format(id))
    player = pd.read_csv(path, index_col=['Start Date'])
    player.index = pd.DatetimeIndex(player.index)
    cols = ['Dis', 'Ct', 'St', 'Ct Wk', 'Ct Fi',
       'Runs_scored', 'Mins', 'BF', '4s', '6s', 'SR', 'Overs', 'Mdns',
       'Runs_concieved', 'Wkts', 'Econ']
    filter_time = (player.index < date)
    filtered_by_time = player[filter_time]
    overall = filtered_by_time[cols].mean().fillna(0).values
    
    filter_team = (player.Opposition == team)
    filtered_by_team = player[filter_team]
    against_team = filtered_by_team[cols].mean().fillna(0).values

    filter_venue = (player.Ground == venue)
    filtered_by_venue = player[filter_venue]
    on_venue = filtered_by_venue[cols].mean().fillna(0).values

    both = filtered_by_team[filter_venue][cols].mean().fillna(0).values

    features = np.hstack((overall, against_team, on_venue, both))
    return features

vread_and_process = np.vectorize(read_and_process,
                                 signature = '(),(),(),()->(n)') 

def recommend_team(players, squad, date, opposition, venue, code):
    # loading model and utils
    clf = load('model.joblib')
    scaler = load('scaler.joblib')
    # Getting ids
    ids = [players.player_id[players.player_name == name].iloc[0] for name in squad.player_name]
    # Prepearing features
    X = vread_and_process(ids, date, opposition, venue)
    # Preprocessing
    X = scaler.transform(X)
    # Predicting
    y = clf.predict_proba(X)[:, 1]
    squad['recommended'] = y
    # Filtering according to code
    batsman = squad[(squad.playing_role == 'Batsman')].sort_values(by = ['recommended'], 
                                                                   ascending = False).head(code[0])
    keeper = squad[(squad.playing_role == 'Wicketkeeper')].sort_values(by = ['recommended'], 
                                                                       ascending = False).head(code[1])
    allrounder = squad[(squad.playing_role == 'Allrounder')].sort_values(by = ['recommended'], 
                                                                         ascending = False).head(code[2])
    bowler = squad[(squad.playing_role == 'Bowler')].sort_values(by = ['recommended'], 
                                                                 ascending = False).head(code[3])
    recommended = pd.concat((batsman, keeper, allrounder, bowler))[['player_name', 'playing_role']]
    print(recommended)
#     print("Chances of win: {:.1f}%".format(squad.recommended[squad.recommended > 0.3].mean() * 100))
    recommended.to_csv('result.csv', index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recmmend team given a squad, opposition team, venue and date")
    parser.add_argument("-t", 
                        help = "Flag to use preset conditions to execute this script",
                        default = 0, type = int)
    parser.add_argument("-d", "--date", 
                        help = "date the match will be played as string in form yyyy-mm-dd",
                        type = str)
    parser.add_argument("-v", "--venue", 
                        help = "Venue for the match",
                        type = str)
    parser.add_argument("-id", "--id_of_players", 
                        help = """ID for players in form 1:2:3 seperated by full colon(:)""",
                        type = str)
    parser.add_argument("-c", "--code", 
                        help = """Describes the scheme of players, 4:1:2:4 corresponds to 4 batsmen,
                        1 wicketkeeper, 2 all rounders and 4 bowler""",
                        type = str)
    parser.add_argument("-o", "--opposition",
                        help = "Team playing agianst",
                        type = str)
    
    args = parser.parse_args()
    
    players_path = os.path.join("Excel Files", "Players_with_not_played_in_matches.csv")
    players = pd.read_csv(players_path)
        
    if args.t == 1:
        squad = players[players.series == 11291]
        print(':'.join(squad.player_id.astype(str).tolist()))
    
        squad = squad.groupby('player_name').head(1).reset_index()[['player_name', 'playing_role']]
        date = np.asarray(['2011-04-05'], dtype = object)
        opposition = 'Sri Lanka'
        venue = 'Sharjah'
        code = [4,1,2,4]
    else:
        ids = args.id_of_players.split(":")
        ids = [int(id) for id in ids]
        mask = players.player_id.map(lambda x: x in ids)
        squad = players[mask]
        squad = squad.groupby('player_name').head(1).reset_index()[['player_name', 'playing_role']]
        date = np.asarray([args.date], dtype = object)
        opposition = args.opposition
        venue = args.venue
        code = [int(num) for num in args.code.split(":")]
        
    
    recommend_team(players, squad, date, opposition, venue, code)