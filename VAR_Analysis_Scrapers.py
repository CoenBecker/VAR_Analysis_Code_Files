## LEAGUE TABLE SCRAPER

# Import libraries
import requests
import pandas as pd
import time
from bs4 import BeautifulSoup
import math
import warnings
import os

# Set directory as e.g. 'C:/Users/username/Documents/folder'
def set_dir():
    while True:
        filedir = input('Set working directory for writing and reading files (e.g. "C:/Users/username/Document/folder"): ')
        if os.path.exists(filedir):
            return filedir
        else:
            print('Please set a valid working directory.')

filedir = set_dir()


# To avoid warnings of deprecated code
warnings.filterwarnings('ignore')

## DEFINE FUNCTIONS FOR THE SCRAPER
# Basic CB measure functions
def gini(pts):
    total = 0
    for i in range(0, len(pts)-1):
        total += sum(abs(pts[i]-pts[i+1:]))
    return total / (len(pts)**2*(sum(pts)/len(pts)))

def herfindahl(sh_pts):
    return sum((sh_pts)**2)

def entropy(shpts):
    info = shpts.apply(lambda x: -math.log2(x) if x > 0 else 0)
    return sum(info*shpts)

# Simple adjustment functions
def hicb(shpts):
    h = sum(shpts**2)
    return (h/(1/len(shpts)))*100

def rentropy(shpts):
    S = entropy(shpts)
    return S/math.log2(len(shpts))

# To scale with min/max functions
# Based on the Most Unequal Distribution
def MUDpoints(n, MP):
    ranks = range(1, n+1)
    return [3*MP[i-1]*((n-ranks[i-1])/(n-1)) for i in ranks]
    # is 3 * matches played * the ratio of opponents that a team in that rank would win against

def MUDshares(n, MP):
    x = MUDpoints(n, MP)
    return x / sum(x)


## SCRAPER

# Dictionary of league number that fbref.com uses, corresponding to each league id used by football-data.co.uk
league_num_dict = {'E0': 9, 'E1': 10, 'E2': 15, 'E3': 16, 'SC0': 40, 'SC1': 72,
                   'D1': 20, 'D2': 33, 'I1': 11, 'I2': 18, 'SP1': 12, 'SP2': 17,
                   'F1': 13, 'F2': 60, 'N1': 23, 'B1': 37, 'P1': 32, 'T1': 26, 'G1': 27}

# Dictionary of when VAR was introduced in each league, based on multiple sources
var_dict = {'E0': 2019, 'E1': 9999, 'E2': 9999, 'E3': 9999, 'SC0': 2022, 'SC1': 9999,
            'D1': 2017, 'D2': 2019, 'F1': 2018, 'F2': 9999, 'I1': 2017, 'I2': 2021,
            'SP1': 2018, 'SP2': 2019, 'N1': 2018, 'B1': 2018, 'G1': 2020, 'T1': 2018, 'P1': 2019}

# The range of seasons to scrape tables for (range(first year 1st season, first year last season + 1))
season_start_yrs = range(2013, 2023)

# A main df to add every scraped table to
df_main = pd.DataFrame()

# Loop through leagues and then through seasons to collect each table
for league_id in league_num_dict:
    print(f"Working on {league_id}")  # Keep track of progress while scraping
    for season_yr in season_start_yrs:
        time.sleep(1)   # To avoid being kicked by the website
        try:
            standings_url = f"https://fbref.com/en/comps/{league_num_dict[league_id]}/{season_yr}-{season_yr + 1}/"
            data = requests.get(standings_url)

            try:
                # This code takes the league results table from the page's html and retains only useful information
                standings_table = pd.read_html(data.text)[0]
                standings_table = standings_table[["Rk", "Squad", "MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]]
                standings_table = standings_table.dropna().reset_index(drop=True)

                # This code adds the season, league ID, and VAR as a variable for the particular table
                standings_table["Season"] = str(season_yr)[2:] + str(season_yr + 1)[2:]
                standings_table["League_id"] = league_id
                if season_yr >= var_dict[league_id]:
                    standings_table['VAR'] = 1
                else:
                    standings_table['VAR'] = 0

                # Now that the core information is gathered, try to add more context from the page's html
                try:
                    soup = BeautifulSoup(data.text)
                    league = soup.select("h1")[0].text  # Returns "20xx-20xx {league name} stats" backended by spaces.
                    league = " ".join(league.split(" ")[1:-1])  # Convert to get the clean league name
                    country = soup.select(".prevnext+ p a")[0].text
                    standings_table["League"] = league
                    standings_table["Country"] = country
                except:
                    print(f"Could not parse soup: {league_id}-{season_yr}")

                # Organize the dataframe
                standings_table = standings_table[["Country", "League", "League_id", "Season", "VAR", "Rk", "Squad",
                                                   "MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]]

                try:
                    # Delete Bury with 0 MP if present
                    idx = standings_table.index[standings_table['MP'] == 0].tolist()[0]
                    standings_table.drop([idx], axis=0, inplace=True)
                    print('Bury was successfully deleted')
                except:
                    pass

                try:
                    # Rename 'Pts' to Orig_pts
                    standings_table['Orig_Pts'] = standings_table['Pts']
                    # Recalculate points in ranking
                    standings_table['Pts'] = standings_table['W'] * 3 + standings_table['D']
                    # Calculate points per match for entropy
                    standings_table['PtsPM'] = standings_table['Pts'] / standings_table['MP']

                    # Calculating different shares of points
                    standings_table['Orig_PtsShare'] = standings_table['Orig_Pts'] / standings_table['Orig_Pts'].sum()
                    standings_table['PPM_Share'] = standings_table['PtsPM'] / sum(standings_table['PtsPM'])
                    standings_table['Sh_pts'] = standings_table['Pts'] / sum(standings_table['Pts'])
                    standings_table['MudShare'] = MUDshares(len(standings_table), standings_table['MP'])

                    # For completeness...
                    standings_table['EqualSh'] = 1 / len(standings_table)
                    standings_table['MudPts'] = MUDpoints(len(standings_table), standings_table['MP'])
                except:
                    print(f"Could not perform point distribution calculations: {league_id}-{season_yr}")

                try:
                    # Calculating base CB values
                    standings_table['Gini'] = gini(standings_table['Pts'])
                    standings_table['HHI'] = herfindahl(standings_table['Sh_pts'])
                    standings_table['Entropy'] = entropy(standings_table['Sh_pts'])

                    # Benchmark original entropy
                    standings_table['Orig_entropy'] = entropy(standings_table['Orig_PtsShare'])

                    ### Do the rest of these on both PPM and normal, see what difference it makes
                    # Calculating proposed n-adjusted CB values
                    standings_table['HICB'] = hicb(standings_table['Sh_pts'])
                    standings_table['R_entropy'] = rentropy(standings_table['Sh_pts'])

                    # MinMax scaled CB values (Scaled from Most Equal Distribution MED=0 to Most Unequal Distr. MUD=1)
                    standings_table['HRCB'] = (herfindahl(standings_table['Sh_pts']) - (1 / len(standings_table))) / (
                            herfindahl(standings_table['MudShare']) - (1 / len(standings_table)))
                    standings_table['CB_entropy'] = (entropy(standings_table['Sh_pts']) - math.log2(len(standings_table))) / (
                            entropy(standings_table['MudShare']) - math.log2(len(standings_table)))
                    standings_table['Adj_gini'] = (gini(standings_table['Sh_pts']) - 0) / (gini(standings_table['MudShare']) - 0)

                    # Calculate CB on Pts per match (PPM)
                    standings_table['PPM_HRCB'] = (herfindahl(standings_table['PPM_Share']) -
                                                   (1 / len(standings_table))) / (herfindahl(standings_table['MudShare']) - (1 / len(standings_table)))
                    standings_table['PPM_CB_entropy'] = (entropy(standings_table['PPM_Share']) - math.log2(len(standings_table))) / (
                            entropy(standings_table['MudShare']) - math.log2(len(standings_table)))
                    standings_table['PPM_Adj_gini'] = (gini(standings_table['PPM_Share']) - 0) / (gini(standings_table['MudShare']) - 0)
                except:
                    print(f"Could not calculate CB values: {league_id}-{season_yr}")

                # Add the data from this league-season observation to the main dataframe
                df_main = pd.concat([df_main, standings_table], ignore_index=True)
            except:
                print(f"Could not get clean table: {league_id}-{season_yr}")
        except:
            print(f"Could not get url: {league_id}-{season_yr}")

df_main.to_csv(f"{filedir}/Rankings_All_LeagueSeasons.csv",
               index=False, encoding="UTF-8")
print("Finished scraping rankings")


###################  MATCH DATA SCRAPER  ############################################

# Import libraries
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup

# STEP 1: GETTING THE MOTHERLIST
base_url = "https://www.football-data.co.uk/"

# Countries for analysis (the name the website uses for this and the omitted leagues is checked in advance)
countries = ['england', 'scotland', 'germany',
             'italy', 'spain', 'france', 'netherlands',
             'belgium', 'portugal', 'turkey', 'greece']

omitted_leagues = ["EC", "SC2", "SC3"]

# Prep lists to add the data about each file(link) scraped for the motherlist
file_list = []
country_list = []
league_list = []

# Loop through every country's page (where links to result-csv's for all its leagues-seasons are listed)
for country in countries:
    page_country = f"{base_url}{country}m.php"
    # Get the html to identify links
    data = requests.get(page_country)
    soup = BeautifulSoup(data.text)

    # Get the block with league names and link to corresponding file of the season
    blocks = soup.find_all('a')

    for block in blocks:
        l = block.get('href')
        if 'mmz4281' in l:           # Only match data files (tagged mmz4281)
            if int(l[8:12]) > 1300 and int(l[8:12]) < 2300:     # Only files of desired seasons (2013-14 (1314) up to 2022-23 (2223))
                if l.split("/")[2].split(".")[0] not in omitted_leagues:     # To exclude omitted leagues (shorthands for Eng Conf, Scottish L. One & Two)
                    # For this file: add to the list, and include the country and league name for reference
                    file_list.append(base_url + l)
                    country_list.append(country.capitalize())
                    league_list.append(block.string)
    time.sleep(2)

# STEP2: GETTING THE FILES AND COMBINING IN THE MAIN DF
df_main = pd.DataFrame()

# Dictionary of when the VAR was introduced, corresponding to the league id football-data.co.uk uses (checked in advance)
var_dict = {'E0': 2019, 'E1': 9999, 'E2': 9999, 'E3': 9999, 'SC0': 2022, 'SC1': 9999,
            'D1': 2017, 'D2': 2019, 'F1': 2018, 'F2': 9999, 'I1': 2017, 'I2': 2021,
            'SP1': 2018, 'SP2': 2019, 'N1': 2018, 'B1': 2018, 'G1': 2020, 'T1': 2018, 'P1': 2019}

# Loop through csv files and get the data
for idx in range(0, len(file_list)):
    url = file_list[idx]
    try:
        df_temp = pd.read_csv(url, encoding="ISO-8859-1")

        df_temp = df_temp.dropna(how='all')  # Drop full n/a rows that occur in the data

        # Add the column containing the indication of the link, country, league, and season of this file
        df_temp['Link'] = url
        df_temp['Country'] = country_list[idx]
        df_temp['League'] = league_list[idx]
        df_temp['Season'] = url.split('/')[4]

        # Calculate desired variables
        df_temp['GD'] = df_temp['FTHG'] - df_temp['FTAG']
        df_temp['ABS_GD'] = abs(df_temp['GD'])

        # Add VAR column
        if int(f"20{df_temp['Season'][0][:2]}") >= int(var_dict[df_temp['Div'][0]]):
            df_temp['VAR'] = 1
        else:
            df_temp['VAR'] = 0

        # Add this df to the main dataframe which will contain all data
        df_main = pd.concat([df_main, df_temp], ignore_index=True)
        time.sleep(1)
    except:
        print(f"Problem with file: {url}")

    # Progress check
    if (idx + 1) % 20 == 10:
        print(f"Treated {idx + 1} files - {len(df_main)} rows - {len(df_main) / (idx + 1)} on average per file")

# Saving the file to csv (full data, for safety)
df_main.to_csv(f"{filedir}/Matches_main_full.csv", index=False, encoding="UTF-8")

# Now clean it up, select only the most relevant columns, and save a clean file
df_main = df_main[["Country", "League", "Div", "Season", "VAR", "Date", "HomeTeam", "AwayTeam",
                   "FTHG", "FTAG", "GD", "ABS_GD", "FTR", "HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
                   "HF", "AF", "HY", "AY", "HR", "AR", "B365H", "B365D", "B365A", "Link"]]
df_main.to_csv(f"{filedir}/Matchdata_main.csv", index=False, encoding="UTF-8")
