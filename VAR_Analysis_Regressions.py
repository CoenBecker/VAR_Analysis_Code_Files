# Import libraries
import pandas as pd
import statsmodels.formula.api as smf
import warnings
import os
from scipy.stats import ttest_ind
from datetime import date

# To avoid warnings of deprecated code
warnings.filterwarnings('ignore')

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

# Import data
matches = pd.read_csv(f'{filedir}/Matches_Analysis.csv')
ragg = pd.read_csv(f'{filedir}/Rankings_Analysis.csv')


# REGRESSION FUNCTIONS


# Basic FE regression with VAR as predictor variable
def regmodelerVAR(y, data):
    # If y contains whitespace, fix
    y_fix = y.replace(' ', '')
    data.rename(columns={y: y_fix}, inplace=True)

    # Define y
    y_name = y_fix

    # Use dummies
    league = "League ID"
    time = "Season"

    # Create a string-values variable of season to avoid patsy error when variable names are composed of only numbers
    data['SeasonString'] = [f"S{t}" for t in data[time]]

    # Get dummies
    league_dummies = pd.get_dummies(data[league])
    time_dummies = pd.get_dummies(data['SeasonString'])
    data = data.join(league_dummies)
    data = data.join(time_dummies)

    # Omit obs with missing dv values
    data = data.dropna(subset=y_name)

    # Define regression equations
    # League and season FE - baseline
    mdl1 = f"{y_name} ~"
    i = 0
    for dumset in [league_dummies[:-1], time_dummies[:-1]]:
        for dummy in dumset.columns[:-1]:
            if i > 0:
                mdl1 = f"{mdl1} + {dummy}"
            else:
                mdl1 = f"{mdl1} {dummy}"
            i += 1

    # Add VAR variable to the model
    mdl2 = f"{y_name} ~ VAR"
    for dumset in [league_dummies[:-1], time_dummies[:-1]]:
        for dummy in dumset.columns[:-1]:
            mdl2 = f"{mdl2} + {dummy}"

    # Run and present models
    for mdl in [mdl1, mdl2]:
        print(mdl)
        model = smf.ols(mdl, data=data)
        model_results = model.fit()
        print(model_results.summary())
        print(" ")
    return


# Difference-in-difference model only on leagues with a solidly defensible control group
def didRobust(dv, df):
    # If y contains whitespace, fix
    dv_fix = dv.replace(' ', '')
    if dv_fix != dv:
        df.rename(columns={dv: dv_fix}, inplace=True)

    # Run model for every country for which strict DiD assumptions can be met
    for country in ['England', 'Scotland', 'France']:

        # Select only data from that country
        data = df[df['Country'] == country]

        # Define TIME variable, to be 1 for all observations from VAR intro year onward, zero otherwise
        var_intro = min(data[data['VAR'] == 1]['Season'])
        data['TIME'] = data['Season'] // var_intro

        # Define TREAT variable, assign 1 to all obs from league that ultimately gets VAR.
        # Which league got VAR
        league_get_var = data[data['VAR'] == 1]['League ID'].unique()

        # Assign 1 if observation is from league that got VAR
        treatgroup = []
        for ID in data['League ID']:
            if ID in league_get_var:
                treatgroup.append(1)
            else:
                treatgroup.append(0)
        data['TREAT'] = treatgroup

        # Define the model
        mdl = f"{dv_fix} ~ TIME + TREAT + TIME:TREAT"

        # Print regression estimations
        print(mdl, f" - {country}")
        model = smf.ols(mdl, data=data)
        model_results = model.fit()
        print(model_results.summary())

        # Leave some space before the next model
        print("\n\n\n\n\n\n")
    return


# Difference-in-difference model with multiple separate treatment groups by common introduction year
def didIntroGroup(dv, df):
    # If dv contains whitespace, fix
    dv_fix = dv.replace(' ', '')
    if dv_fix != dv:
        df.rename(columns={dv: dv_fix}, inplace=True)

    # Find all treatment groups by common introduction year, and control group as all that never got VAR
    # Aggregate match data to find average VAR value per league
    maggID = matches.groupby(['League ID'], as_index=False).mean()

    # Combine all league IDs that never got VAR in control group (mean VAR = 0 if never introduced)
    control = [x for x in maggID[maggID['VAR'] == 0]['League ID']]

    # Got VAR first
    # Get list of all league-seasons where VAR was present
    magg = matches.groupby(['League ID', 'Season'], as_index=False).mean()
    maggVAR = magg[magg['VAR'] == 1]

    # Get list of all leagues that had it in 1718, they are the first to introduce
    var1718 = [x for x in maggVAR[maggVAR['Season'] == 1718]['League ID']]
    # Remove these leagues from data to avoid double counting
    maggVAR = maggVAR[-maggVAR['League ID'].isin(var1718)]

    # Repeat, now find all who had it in 1819, since they are in the list, it must be their first year
    var1819 = [x for x in maggVAR[maggVAR['Season'] == 1819]['League ID']]
    maggVAR = maggVAR[-maggVAR['League ID'].isin(var1819)]

    # Repeat for remaining years
    var1920 = [x for x in maggVAR[maggVAR['Season'] == 1920]['League ID']]
    maggVAR = maggVAR[-maggVAR['League ID'].isin(var1920)]

    var2021 = [x for x in maggVAR[maggVAR['Season'] == 2021]['League ID']]
    maggVAR = maggVAR[-maggVAR['League ID'].isin(var2021)]

    var2122 = [x for x in maggVAR[maggVAR['Season'] == 2122]['League ID']]
    maggVAR = maggVAR[-maggVAR['League ID'].isin(var2122)]

    var2223 = [x for x in maggVAR[maggVAR['Season'] == 2223]['League ID']]

    # Run model for every VAR intro group, against the combined control group leagues
    for intro_group in [var1718, var1819, var1920, var2021, var2122, var2223]:
        # Select all data from leagues in this treatment group
        treated = matches[matches['League ID'].isin(intro_group)]

        # Select control group data
        controls = matches[matches['League ID'].isin(control)]

        # Combine data from treat and control in one frame
        data = pd.concat([treated, controls])

        # Define TIME variable, to be 1 for all observations from VAR intro year onward, zero otherwise
        var_intro = min(data[data['VAR'] == 1]['Season'])
        data['TIME'] = data['Season'] // var_intro  # Returns zero before intro year, 1 during and after

        # Define TREAT variable, assign 1 to all obs from league that ultimately gets VAR.
        treatgroup = []
        for ID in data['League ID']:
            if ID in intro_group:
                treatgroup.append(1)
            else:
                treatgroup.append(0)
        data['TREAT'] = treatgroup

        # Define the model
        mdl = f"{dv_fix} ~ TIME + TREAT + TIME:TREAT"

        # Print sample details to keep track of current treatment/control leagues
        print(var_intro)  # To keep track of which treatment group it is
        print(f"Treatment: {treated['League ID'].unique()}")
        print(f"Controls: {controls['League ID'].unique()}")

        # Print regression results
        print(mdl)
        model = smf.ols(mdl, data=data)
        model_results = model.fit()
        print(model_results.summary())

        # Leave some space before the next model
        print("\n\n\n\n\n\n\n\n")
    return

# RUN ANALYSES

## FE regression - probabilities entropy, abs spread, nom spread, draw prob.
for dv in ['Prob_Entropy', 'Absolute Spread Home_Away', 'Nominal Spread Home_Away', 'Draw Probability']:
    regmodelerVAR(dv, matches)

## FE regression - CB - HRCB, Entropy Ratio, Gini Ratio
for dv in ['PPM HRCB', 'PPM Entropy Ratio', 'PPM Gini Ratio']:
    regmodelerVAR(dv, ragg)

## DiD by selected countries - robust treatment/control groups
for dv in ['Prob_Entropy', 'Absolute Spread Home_Away']:
    didRobust(dv, matches)

## DiD by intro_year
didIntroGroup('Prob_Entropy', matches)


# FURTHER ANALYSES

# T-test by league function
def ttest_by_leagueID(y, df):
    res = []
    for var in y:
        # For every league in the sample separately
        for c in df['League ID'].unique():
            dft = df[df['League ID'] == c]
            df_pre = dft[dft['VAR'] == 0]
            df_post = dft[dft['VAR'] == 1]
            pre = df_pre[var].dropna()
            post = df_post[var].dropna()
            stat, pval = ttest_ind(post, pre, equal_var=False)
            if pval < 0.1:
                sig = '*'
                if pval < 0.05:
                    sig = '**'
                    if pval < 0.01:
                        sig = '***'
            else:
                sig = ''
            mean_diff = post.mean() - pre.mean()
            mean_d_perc = f"{round((mean_diff / pre.mean()) * 100, 2)} %"
            if len(y) == 1:
                row = [c, len(pre), pre.mean(), len(post), post.mean(), mean_d_perc, stat, pval, sig]
            else:
                row = [c, var, len(pre), pre.mean(), len(post), post.mean(), mean_d_perc, stat, pval, sig]
            res.append(row)
    if len(y) == 1:
        results_ttest = pd.DataFrame(res, columns=['League', 'nPre', 'Pre-mean', 'nPost', 'Post-mean',
                                                   'Diff %', 'T-statistic', 'P-value', 'Significance'])
    else:
        results_ttest = pd.DataFrame(res, columns=['League', 'Variable', 'nPre', 'Pre-mean', 'nPost', 'Post-mean',
                                                   'Diff %', 'T-statistic', 'P-value', 'Significance'])

    # Save results to file
    today = date.today()
    results_ttest.round(decimals=3).to_csv(f"{filedir}/VAR_Analysis_DVbyLeague_{today}.csv", index=False)

    return results_ttest.round(decimals=3)


# Redefine the select seasons before and after VAR for a balanced sample for t-tests
# Redefine when VAR is introduced in each league
var_dict = {'E0': 2019, 'E1': 9999, 'E2': 9999, 'E3': 9999, 'SC0': 2022, 'SC1': 9999,
            'D1': 2017, 'D2': 2019, 'F1': 2018, 'F2': 9999, 'I1': 2017, 'I2': 2021,
            'SP1': 2018, 'SP2': 2019, 'N1': 2018, 'G1': 2020, 'T1': 2018, 'P1': 2019}

# Create a dictionary with the first and last season to take from each league that got VAR
selections = dict()
for key in var_dict:  # For every league
    if var_dict[key] < 9999:  # If league got VAR
        start_var = int(str(var_dict[key])[2:])
        n_var_seasons = 23 - start_var  # n seasons in sample with VAR
        n_novar_seasons = start_var - 13  # n seasons in sample without VAR
        select_n = 2 * min(n_var_seasons,
                           n_novar_seasons)  # Get the smallest, multiply by 2 to get total n seasons to select
        select_start = str(int(start_var - select_n / 2)) + str(
            int(start_var - select_n / 2 + 1))  # First season to select
        select_end = str(int(start_var + select_n / 2 - 1)) + str(
            int(start_var + select_n / 2))  # Last season to select
        # Add start/end season to dictionary
        selections[key] = [int(select_start), int(select_end)]

# Select matches from league and only seasons in selections dictionary
select_matches = pd.DataFrame()
for key in selections:
    frame = matches[matches['League ID'] == key]
    frame = frame[frame['Season'] >= selections[key][0]]
    frame = frame[frame['Season'] <= selections[key][1]]
    select_matches = pd.concat([select_matches, frame])


# Run t-test by league
ttest_by_leagueID(['Prob_Entropy'], select_matches)


# DiD by league vs combined control group
maggID = matches.groupby(['League ID'], as_index=False).mean()
control = [x for x in maggID[maggID['VAR'] == 0]['League ID']]
treated = [x for x in maggID[maggID['VAR'] != 0]['League ID']]


# Define new function
def didByLeague(dv, df):
    # If dv contains whitespace, fix
    dv_fix = dv.replace(' ', '')
    if dv_fix != dv:
        df.rename(columns={dv: dv_fix}, inplace=True)

    for league in treated:
        treat = df[df['League ID'] == league]
        ctrl = df[df['League ID'].isin(control)]
        data = pd.concat([treat, ctrl])
        var_intro = min(treat[treat['VAR'] == 1]['Season'])
        data['TIME'] = data['Season'] // var_intro

        treatval = []
        for ID in data['League ID']:
            if ID in treated:
                treatval.append(1)
            else:
                treatval.append(0)
        data['TREAT'] = treatval

        mdl = F"{dv_fix} ~ TIME + TREAT + TIME:TREAT"

        print(f"VAR intro: {var_intro}")
        print(f"Treatment: {treat['League ID'].unique()}")
        print(f"Controls: {ctrl['League ID'].unique()}")
        print(mdl)

        model = smf.ols(mdl, data=data)
        model_results = model.fit()
        print(model_results.summary())
        print(" ")
    return

# Run the DiD by league model on Probabilities' entropy
didByLeague('Prob_Entropy', matches)


# Separate analysis by high/low league CB
# Reload clean files
matches = pd.read_csv(f'{filedir}/Matches_Analysis.csv')
ragg = pd.read_csv(f'{filedir}/Rankings_Analysis.csv')

aggLeagues = ragg.groupby(['League ID'], as_index=False).mean()
# aggLeagues['PPM HRCB'].describe()  # To get a sense of where to split

# Split sample into high or low 50/50 by median (50th percentile)
lowCBleagues = aggLeagues[aggLeagues['PPM HRCB'] < aggLeagues['PPM HRCB'].quantile([0.5])[0.5]]['League ID']
highCBleagues = aggLeagues[aggLeagues['PPM HRCB'] > aggLeagues['PPM HRCB'].quantile([0.5])[0.5]]['League ID']

# See which leagues are in the high and low CB groups
print('Low CB leagues', list(lowCBleagues), '\nHigh CB leagues', list(highCBleagues))

# Get the subset of match data for each high/low CB group
highCBmatch = matches[matches['League ID'].isin(highCBleagues)]
lowCBmatch = matches[matches['League ID'].isin(lowCBleagues)]

# Get the descriptives of each group's match data (the variables under investigation)
lowCBdesc = pd.DataFrame(lowCBmatch[['VAR', 'Prob_Entropy', 'Absolute Spread Home_Away', 'Nominal Spread Home_Away',
                                     'Home Probability', 'Draw Probability', 'Away Probability']].describe())
highCBdesc = pd.DataFrame(highCBmatch[['VAR', 'Prob_Entropy', 'Absolute Spread Home_Away', 'Nominal Spread Home_Away',
                                       'Home Probability', 'Draw Probability', 'Away Probability']].describe())
# Save descriptives to file
with pd.ExcelWriter(f"{filedir}/VAR_Analysis_HighLowCB_Descriptives.xlsx") as writer:
    # To store the dataframe in specified sheet
    lowCBdesc.to_excel(writer, sheet_name='LowCB_desc')
    highCBdesc.to_excel(writer, sheet_name='HighCB_desc')

# Regressions on high/low CB groups
for dv in ['Prob_Entropy', 'Absolute Spread Home_Away', 'Nominal Spread Home_Away',
           'Home Probability', 'Draw Probability', 'Away Probability']:
    print('Low CB panel analyses:')
    regmodelerVAR(dv, lowCBmatch)

for dv in ['Prob_Entropy', 'Absolute Spread Home_Away', 'Nominal Spread Home_Away',
           'Home Probability', 'Draw Probability', 'Away Probability']:
    print('High CB panel analyses')
    regmodelerVAR(dv, highCBmatch)


# Addition of regression with HRCB as a moderator

# Define the function
def regmodeler(data, y, xvar=['VAR'], c=False):
    # if y contains whitespace, fix
    y_fix = y.replace(' ', '')
    data.rename(columns={y: y_fix}, inplace=True)
    # if xvar contains whitespace
    for x in xvar:
        if x != x.replace(' ', ''):
            x_fix = x.replace(' ', '')
            data = data.rename(columns={x: x_fix})
            xvar.remove(x)
            xvar.append(x_fix)
    # if an interaction term is included of variables with spaces
    if c != False:
        for ivar in c:
            if ivar != ivar.replace(' ', ''):
                ivar_fix = ivar.replace(' ', '')
                data = data.rename(columns={ivar: ivar_fix})
                c.remove(ivar)
                c.append(ivar_fix)

    # Define x and y
    y_name = y_fix
    x_names = xvar

    # Define dummy source variables
    league = "League ID"
    time = "Season"

    # Create a string-values variable of season to avoid reg. equation error when variable names are composed of only numbers
    data['SeasonString'] = [f"S{t}" for t in data[time]]

    # Get dummies
    league_dummies = pd.get_dummies(data[league])
    time_dummies = pd.get_dummies(data['SeasonString'])
    data = data.join(league_dummies)
    data = data.join(time_dummies)

    # omit obs with missing dv values
    data = data.dropna(subset=y_name)

    # define regression equations
    # league and season FE - baseline level 2
    mdl1 = f"{y_name} ~"
    i = 0
    for dumset in [league_dummies[:-1], time_dummies[:-1]]:
        for dummy in dumset.columns[:-1]:
            if i > 0:
                mdl1 = f"{mdl1} + {dummy}"
            else:
                mdl1 = f"{mdl1} {dummy}"
            i += 1

    # Add in VAR (or other main IV)
    mdl2 = f"{mdl1} + {x_names[0]}"

    # Add in other IVs
    mdl3 = f"{mdl2}"
    for var in x_names[1:]:
        mdl3 = f"{mdl3} + {var}"

    # Add in interaction
    mdl4 = False  # default
    if c != False:
        mdl4 = f"{mdl3} + {c[0]}:{c[1]}"

    # Run and present models
    mdl_list = [mdl1, mdl2, mdl3]
    if mdl4 != False:
        mdl_list.append(mdl4)

    for mdl in mdl_list:
        print(mdl)
        model = smf.ols(mdl, data=data)
        model_results = model.fit()
        print(model_results.summary())
        print(" ")

    return


# Add HRCB to match data

# Convert season to string and concat with league id to get unique league-season identifiers
matches['SeasonString'] = [str(x) for x in matches['Season']]
matches['LeagueSeason'] = matches['League ID'] + matches['SeasonString']

# Do the same to get league-season identifier in rankings data
ragg['SeasonString'] = [str(x) for x in ragg['Season']]
ragg['LeagueSeason'] = ragg['League ID'] + ragg['SeasonString']

# To check if the number of unique league-seasons in matches data is the same as in ragg for the dictionary
print(len(matches['LeagueSeason'].unique()) == len(ragg['LeagueSeason'].unique()))

# Create a dictionary with the values of CB (by HRCB) for each league-season in aggregated rankings data.
LS_CB_dict = dict()
for i in range(0, len(ragg['LeagueSeason'])):
    LS_CB_dict[ragg['LeagueSeason'][i]] = ragg['PPM HRCB'][i]

# Assign HRCB values to each match observation
matches['HRCB'] = [LS_CB_dict[i] for i in matches['LeagueSeason']]

# Check to see if it is done correctly
magg = matches.groupby(['LeagueSeason'], as_index=False).mean().sort_values(by='LeagueSeason')
ragg = ragg.sort_values(by='LeagueSeason')
print(list(magg['HRCB'].round(10)) == list(ragg['PPM HRCB'].round(10)))

# Run the model
for dv in ['Prob_Entropy', 'Nominal Spread Home_Away']:
    regmodeler(matches, dv, ['VAR', 'HRCB'], ['VAR', 'HRCB'])
