# CLEAN RANKINGS: REMOVE TEAMS WITH >5 INVALID MATCHES IN ONE SEASON

# Import packages
import pandas as pd
import math
import numpy as np
import os

# Set directory as e.g. 'C:/Users/username/Documents/folder'
def set_dir():
    while True:
        filedir = input('Set working directory for writing and reading files (e.g. "C:/Users/username/Document/folder"): ')
        if os.path.exists(filedir):
            return filedir
        else:
            print('Please set a valid working directory.')

# Check if filedir is still set from previous code
try:
    os.path.exists(filedir)
except:
    filedir = set_dir()

# Import match data to identify teams with too many invalid matches, to treat rankings
matches = pd.read_csv(f"{filedir}/Matchdata_main.csv")

# Take the HT missing matches
matches_invalid = matches[matches['HTHG'].isnull()]

# Problem teams: take those teams with >5 invalid matches in a season, to remove these teams.
prob_home = matches_invalid[['Div', 'Season', 'HomeTeam']]
prob_home = prob_home.rename(columns={'HomeTeam': 'Team'})
prob_away = matches_invalid[['Div', 'Season', 'AwayTeam']]
prob_away = prob_away.rename(columns={'AwayTeam': 'Team'})
prob_teams = pd.concat([prob_home, prob_away])
prob_teams['Count'] = 1
prob_teams = prob_teams.groupby(['Div', 'Season', 'Team']).count()
prob_teams = prob_teams[prob_teams['Count'] > 5]

# To check which leagues they are in
## prob_leagues = [[i[0], i[1]] for i in prob_teams.index]

# Inspect the teams
prob_teamnames = [i[2] for i in prob_teams.index]
print(prob_teamnames)
# Manually inspect what these teams are called in the rankings' data set and create new list:
prob_teamnames = ['Niki Volos', 'OFI Crete', 'Reus', 'Gaziantep FK', 'Hatayspor']


# Redefine functions
def gini(pts):
    total = 0
    for i in range(0, len(pts) - 1):
        total += sum(abs(pts[i] - pts[i + 1:]))
    return total / (len(pts) ** 2 * (sum(pts) / len(pts)))


def herfindahl(shpts):
    return sum(shpts ** 2)


def entropy(shpts):
    import math
    info = shpts.apply(lambda x: -math.log2(x) if x > 0 else 0)
    return sum(info * shpts)


def hicb(shpts):
    h = sum(shpts ** 2)
    return (h / (1 / len(shpts))) * 100


def rentropy(shpts):
    import math
    s = entropy(shpts)
    return s / math.log2(len(shpts))


def MUDpoints(n, mp):
    ranks = range(1, n + 1)
    return [3 * mp[i - 1] * ((n - ranks[i - 1]) / (n - 1)) for i in ranks]


def MUDshares(n, mp):
    x = MUDpoints(n, mp)
    return x / sum(x)


# Take the table from each of the league-seasons with a problematic team, redo table, then reset in rankings
rankings = pd.read_csv(f"{filedir}/Rankings_All_LeagueSeasons.csv")

# Code to treat tables
treated = []
for i in prob_teams.index:
    league_season = [i[0], i[1]]
    if league_season not in treated:
        # Get table from rankings if it is not already treated
        table = rankings[rankings['League_id'] == i[0]]
        table = table[table['Season'] == i[1]]

        # Drop this old table from the rankings' data set
        rankings = pd.concat([rankings, table]).drop_duplicates(keep=False)

        # Now remove the problematic teams from the table
        table = table[table.Squad.isin(prob_teamnames) == False].reset_index(drop=True)

        # Recalculate values for table
        # Reset rank
        table['Rk'] = np.arange(1, len(table) + 1)
        # Share of points
        table['Orig_PtsShare'] = table['Orig_Pts'] / table['Orig_Pts'].sum()
        table['PPM_Share'] = table['PtsPM'] / sum(table['PtsPM'])
        table['Sh_pts'] = table['Pts'] / sum(table['Pts'])
        table['MudShare'] = MUDshares(len(table), table['MP'])
        table['EqualSh'] = 1 / len(table)
        table['MudPts'] = MUDpoints(len(table), table['MP'])

        # CB measures
        table['Gini'] = gini(table['Pts'])
        table['HHI'] = herfindahl(table['Sh_pts'])
        table['Entropy'] = entropy(table['Sh_pts'])
        table['Orig_entropy'] = entropy(table['Orig_PtsShare'])
        table['HICB'] = hicb(table['Sh_pts'])
        table['R_entropy'] = rentropy(table['Sh_pts'])
        table['HRCB'] = (herfindahl(table['Sh_pts']) - (1 / len(table))) / (
                herfindahl(table['MudShare']) - (1 / len(table)))
        table['CB_entropy'] = (entropy(table['Sh_pts']) - math.log2(len(table))) / (
                entropy(table['MudShare']) - math.log2(len(table)))
        table['Adj_gini'] = (gini(table['Sh_pts']) - 0) / (gini(table['MudShare']) - 0)
        table['PPM_HRCB'] = (herfindahl(table['PPM_Share']) - (1 / len(table))) / (
                herfindahl(table['MudShare']) - (1 / len(table)))
        table['PPM_CB_entropy'] = (entropy(table['PPM_Share']) - math.log2(len(table))) / (
                entropy(table['MudShare']) - math.log2(len(table)))
        table['PPM_Adj_gini'] = (gini(table['PPM_Share']) - 0) / (gini(table['MudShare']) - 0)

        # Insert table back in rankings
        rankings = pd.concat([rankings, table], ignore_index=True)

        # Record the treatment of this season
        treated.append(league_season)

# Re-order rankings by Country, League_id, Season, Rank
rankings.sort_values(by=['Country', 'League_id', 'Season', 'Rk'], ignore_index=True, inplace=True)

rankings.to_csv(f"{filedir}/Rankings_Clean.csv", index=False)


## PREP A RANKINGS FOR ANALYSIS FILE
rankings1 = pd.read_csv(f"{filedir}/Rankings_Clean.csv")

# (de)select all Belgian tables (proved to be a problematic league)
rankingsBel = rankings1[rankings1['Country'] == 'Belgium']
rankings = pd.concat([rankings1, rankingsBel]).drop_duplicates(keep=False)

# Rename columns for readability
rankings.columns = ['Country', 'League', 'League ID', 'Season', 'VAR', 'Rank', 'Team', 'Matches Played',
                    'Wins', 'Draws', 'Losses', 'Goals For', 'Goals Against', 'Goal Difference', 'Points',
                    'Orig. Points', 'Points per Match', 'Orig. Points Share', 'PPM Share',
                    'Points Share', 'MUD Share', 'Equal Shares', 'MUD Points',
                    'Gini', 'HHI', 'Entropy', 'Orig. Entropy', 'HICB', 'R Entropy', 'HRCB',
                    'Entropy Ratio', 'Gini Ratio', 'PPM HRCB', 'PPM Entropy Ratio', 'PPM Gini Ratio']

# Save the cleaned file
rankings.to_csv(f"{filedir}/Rankings_Full_Analysis.csv", index=False)

# To aggregate the rankings' data to the 190 league-season obs
ragg = rankings.drop(["Country", "League", "Team"], axis=1)
ragg = ragg.groupby(["League ID", "Season"], as_index=False).mean()

## Prep a Ranking_Analysis file with only most relevant variables and which is aggregated at the league-season level
ragg = ragg[['League ID', 'Season', 'VAR', 'Rank', 'Matches Played', 'Wins', 'Draws', 'Losses',
             'Goals For', 'Goals Against', 'Goal Difference', 'Points', 'Points per Match',
             'PPM Share', 'Points Share', 'MUD Share', 'Gini', 'HHI', 'Entropy', 'HICB', 'R Entropy',
             'HRCB', 'Entropy Ratio', 'Gini Ratio', 'PPM HRCB', 'PPM Entropy Ratio', 'PPM Gini Ratio']]
# Save file
ragg.to_csv(f"{filedir}/Rankings_Analysis.csv", index=False)


## CLEAN MATCH DATA
matches = pd.read_csv(f"{filedir}/Matchdata_main.csv")

# Matchdata can be fully resolved just by treating the first dataset
# (de)select all matches missing HT scores (confirmed to all be problematic)
matchesHTna = matches[matches['HTHG'].isnull()]
matchesValid = pd.concat([matches, matchesHTna]).drop_duplicates(keep=False)

# (de)select all matches from the Belgian league (turned out to be problematic)
matchesBel = matchesValid[matchesValid['Country'] == 'Belgium']
matchesClean = pd.concat([matchesValid, matchesBel]).drop_duplicates(keep=False)
matchesClean.to_csv(f"{filedir}/MatchesClean.csv", index=False)

## Clean up names and add columns
matches = pd.read_csv(f"{filedir}/MatchesClean.csv")
matches.columns = ['Country', 'League', 'League ID', 'Season', 'VAR', 'Date', 'Home Team',
                   'Away Team', 'Full-Time Home Goals', 'Full-Time Away Goals', 'Goal Difference',
                   'Absolute Goal Difference', 'Full-Time Result', 'Half-Time Home Goals', 'Half-Time Away Goals',
                   'Half-Time Result', 'Home Shots', 'Away Shots', 'Home Shots on Target', 'Away Shots on Target',
                   'Home Fouls', 'Away Fouls', 'Home Yellow Cards', 'Away Yellow Cards', 'Home Red Cards',
                   'Away Red Cards', 'B365 Home Odds', 'B365 Draw Odds', 'B365 Away Odds', 'Link']

matches['Goals'] = matches['Full-Time Home Goals'] + matches['Full-Time Away Goals']
matches['Shots'] = matches['Home Shots'] + matches['Away Shots']
matches['Shots on Target'] = matches['Home Shots on Target'] + matches['Away Shots on Target']
matches['Fouls'] = matches['Home Fouls'] + matches['Away Fouls']
matches['Yellow Cards'] = matches['Home Yellow Cards'] + matches['Away Yellow Cards']
matches['Red Cards'] = matches['Home Red Cards'] + matches['Away Red Cards']
matches['Cards'] = matches['Yellow Cards'] + matches['Red Cards']

# Organize the df
matches = matches[['Country', 'League', 'League ID', 'Season', 'VAR', 'Date', 'Home Team', 'Away Team',
                   'Goals', 'Full-Time Home Goals', 'Full-Time Away Goals', 'Goal Difference',
                   'Absolute Goal Difference', 'Full-Time Result', 'Half-Time Home Goals', 'Half-Time Away Goals',
                   'Half-Time Result', 'Shots', 'Home Shots', 'Away Shots', 'Shots on Target',
                   'Home Shots on Target', 'Away Shots on Target', 'Fouls', 'Home Fouls', 'Away Fouls', 'Cards',
                   'Yellow Cards', 'Home Yellow Cards', 'Away Yellow Cards', 'Red Cards', 'Home Red Cards',
                   'Away Red Cards', 'B365 Home Odds', 'B365 Draw Odds', 'B365 Away Odds', 'Link']]

# Calculate probabilities from odds
matches['Home Probability'] = 1 / matches['B365 Home Odds']
matches['Draw Probability'] = 1 / matches['B365 Draw Odds']
matches['Away Probability'] = 1 / matches['B365 Away Odds']

# Temporary total probability to divide probabilities by
matches['total_pr'] = matches['Home Probability'] + matches['Draw Probability'] + matches['Away Probability']

# Adjust for betting office's margin
matches['Home Probability'] = matches['Home Probability'] / matches['total_pr']
matches['Draw Probability'] = matches['Draw Probability'] / matches['total_pr']
matches['Away Probability'] = matches['Away Probability'] / matches['total_pr']

# Drop temporary variable
matches.drop('total_pr', axis=1, inplace=True)

# Create a spread variable for the difference between home and away win probability
matches['Nominal Spread Home_Away'] = matches['Home Probability'] - matches['Away Probability']
# And its absolute value
matches['Absolute Spread Home_Away'] = [abs(x) for x in matches['Nominal Spread Home_Away']]

# Define an entropy for match probabilities UO function
def prob_entropy(probs):
    try:
        wgt_info = [math.log2(x)*x for x in probs]
        return -sum(wgt_info)
    except:
        return math.nan

# Calculate the entropy for each set of match outcome probabilities
prob_entropies = []
for index, row in matches.iterrows():
    prob_entropies.append(prob_entropy(row[['Home Probability', 'Draw Probability', 'Away Probability']]))

# Create a column for the probability entropy
matches['Prob_Entropy'] = prob_entropies

# Save this file for analyses
matches.to_csv(f"{filedir}/Matches_Analysis.csv", index=False)
