# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from datetime import date
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel


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
matches = pd.read_csv(f'{filedir}/Matches_Analysis.csv')  # Matches
rankings = pd.read_csv(f'{filedir}/Rankings_Full_Analysis.csv')  # Rankings in full (all team-entries)
ragg = pd.read_csv(f'{filedir}/Rankings_Analysis.csv')  # Rankings aggregated at league-level


# DESCRIPTIVE STATISTICS
# Get basic descriptives for match data
mdesc = pd.DataFrame(matches.describe())

# Find numeric variables and get their skewness and kurtosis in a df
mnum = matches.select_dtypes(include='number')
mskurt = pd.DataFrame([mnum.skew(), mnum.kurt()], index=['skew', 'kurt'])

# Add skewness and kurtosis to the basic descriptives dataframe
mdesc = pd.concat([mdesc, mskurt], ignore_index=False)


# Repeat this for rankings data (with full team-entries)
rdesc = pd.DataFrame(rankings.describe())
rnum = rankings.select_dtypes(include='number')
rskurt = pd.DataFrame([rnum.skew(), rnum.kurt()], index=['skew', 'kurt'])
rdesc = pd.concat([rdesc, rskurt], ignore_index=False)

# Repeat it for aggregated rank data (to get appropriate level for league-level variables)
rdesc_agg = pd.DataFrame(ragg.describe())
rnum = ragg.select_dtypes(include='number')
rskurt = pd.DataFrame([rnum.skew(), rnum.kurt()], index=['skew', 'kurt'])
rdesc_agg = pd.concat([rdesc_agg, rskurt], ignore_index=False)


# CORRELATIONS
# Get match data correlations from numeric variables
mcorr = matches.select_dtypes(include='number')
rho = mcorr.corr()

# Get p-values and convert to *,**,***
pval = mcorr.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.apply(lambda x: x.apply(lambda xx: ''.join(['*' for t in [.05, .01, .001] if xx <= t])))

# Combine correlation coefficients and their signficance indications
mcorr = rho.round(3).astype(str) + p


# Repeat to retrieve rankings data correlations
rcorr = rankings.select_dtypes(include='number')
rho = rcorr.corr()
pval = rcorr.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.apply(lambda x: x.apply(lambda xx: ''.join(['*' for t in [.05, .01, .001] if xx <= t])))
rcorr = rho.round(3).astype(str) + p

# Repeat to retrieve aggregated rankings data correlations
rcorr_agg = ragg.select_dtypes(include='number')
rho = rcorr_agg.corr()
pval = rcorr_agg.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.apply(lambda x: x.apply(lambda xx: ''.join(['*' for t in [.05, .01, .001] if xx <= t])))
rcorr_agg = rho.round(3).astype(str) + p


# To name the file according to date (if ran again, it will not return an error for file already existing)
today = date.today()

# Create a excel writer object and write each data-level's descriptives and correlations to separate sheets
with pd.ExcelWriter(f"{filedir}/VAR_Analysis_Descriptives_{today}.xlsx") as writer:
    # To store the dataframe in specified sheet
    mdesc.to_excel(writer, sheet_name="Match_desc")
    rdesc.to_excel(writer, sheet_name="Rank_desc")
    mcorr.to_excel(writer, sheet_name="Match_corr")
    rcorr.to_excel(writer, sheet_name="Rank_corr")
    rdesc_agg.to_excel(writer, sheet_name="Rank_agg_desc")
    rcorr_agg.to_excel(writer, sheet_name="Rank_agg_corr")


# GET BALANCED SAMPLES FOR T-TESTS AND HISTOGRAMS
# Select only leagues that got VAR and an equal number of seasons before and after its introduction in the league

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

# Select rankings data league-seasons that are in the dictionary
select_rankings = pd.DataFrame()
for key in selections:
    frame = rankings[rankings['League ID'] == key]
    frame = frame[frame['Season'] >= selections[key][0]]
    frame = frame[frame['Season'] <= selections[key][1]]
    select_rankings = pd.concat([select_rankings, frame])

# Repeat for aggregated rankings data
select_ragg = pd.DataFrame()
for key in selections:
    frame = ragg[ragg['League ID'] == key]
    frame = frame[frame['Season'] >= selections[key][0]]
    frame = frame[frame['Season'] <= selections[key][1]]
    select_ragg = pd.concat([select_ragg, frame])


# VISUALIZATION

# Subset from the dataframes only the variables needed in visualizations
matches_vis = select_matches[['VAR', 'Draw Probability', 'Nominal Spread Home_Away', 'Absolute Spread Home_Away', 'Prob_Entropy']]
ragg_vis = select_ragg[['VAR', 'PPM HRCB', 'PPM Entropy Ratio', 'PPM Gini Ratio']]

# Rename for appropriate formatting on figures
matches_vis.rename(columns={'Nominal Spread Home_Away': 'Nom. Spread HomeAway',
                            'Absolute Spread Home_Away': 'Abs. Spread HomeAway',
                            'Prob_Entropy': 'Probabilities Entropy'}, inplace=True)
ragg_vis.rename(columns={'PPM HRCB': 'HRCB', 'PPM Entropy Ratio': 'Entropy Ratio', 'PPM Gini Ratio': 'Gini Ratio'},
                inplace=True)

# Create a writer object to save figures to excel
writer = pd.ExcelWriter(f"{filedir}/VAR_Analysis_Visualizations_{today}.xlsx", engine='xlsxwriter')
workbook = writer.book

# Define separate sheets for saving match- and rankings-based figures
worksheet1 = workbook.add_worksheet('Match Variables')
worksheet2 = workbook.add_worksheet('Ragg Variables')

# Set counter for numbering/naming figure png files
count = 1

# GENERATE MATCH VARIABLES VISUALISATIONS
# Set index to indicate where in sheet to paste a figure, to add fitting intervals to
idx = 2

# Generate histogram combining before and after VAR distributions of selected variables
for dep_var in ['Draw Probability', 'Nom. Spread HomeAway', 'Abs. Spread HomeAway', 'Probabilities Entropy']:
    sns.displot(matches_vis, x=dep_var, hue='VAR', alpha=0.5, palette='pastel', bins=15, linewidth=0.1,
                stat='probability', common_norm=False, kde=True, kde_kws={'bw_adjust': 5})
    plt.savefig(f'VAR_A_Fig{count}.png')
    plt.close()
    worksheet1.insert_image(f'B{idx}', f'VAR_A_Fig{count}.png')
    count += 1
    idx += 30

# RAGG VISUALISATIONS
# Reset index to start at the top of the new sheet, don't reset the count for numbering png files
idx = 2

# Generate combined before/after VAR histograms for selected variables
for dep_var in ['HRCB', 'Entropy Ratio', 'Gini Ratio']:
    sns.displot(ragg_vis, x=dep_var, hue='VAR', alpha=0.5, palette='pastel', bins=25, linewidth=0.1,
                stat='probability', common_norm=False, kde=True, kde_kws={'bw_adjust': 5})
    plt.savefig(f'VAR_A_Fig{count}.png')
    plt.close()
    worksheet2.insert_image(f'B{idx}', f'VAR_A_Fig{count}.png')
    count += 1
    idx += 30

writer.save()


# T-TESTS
# Define the dfs to do t-tests on
dataframes = [select_matches, select_rankings, select_ragg]

# Empty list to add results of each t-test to
res = []

# Counter to track which df is tested (to use welch on all but the league-level variables in aggregated rankings file)
df_test = 0

# Loop over all dfs and split sample into before and after VAR introduction
for df in dataframes:
    df_test += 1
    df_pre = df[df['VAR'] == 0]
    df_post = df[df['VAR'] == 1]

    # Run t-tests for all numeric variables, except a few meaningless ones
    for var in df.select_dtypes(include='number'):
        if var in ['Season', 'VAR', 'Rank', 'League ID']:
            pass
        else:
            pre = df_pre[var].dropna()
            post = df_post[var].dropna()
            if df_test == 3:
                stat, pval = ttest_rel(pre, post)  # use a paired samples t-test on the aggregated rankings file
            else:
                stat, pval = ttest_ind(post, pre, equal_var=False)  # use a welch t-test for every other test

            # Add significance
            if pval < 0.1:
                sig = '*'
                if pval < 0.05:
                    sig = '**'
                    if pval < 0.01:
                        sig = '***'
            else:
                sig = ''

            # Find difference between before/after VAR means as a percentage
            mean_diff = post.mean() - pre.mean()
            mean_d_perc = f"{round((mean_diff / pre.mean()) * 100, 2)} %"

            # Add the results to the results list
            row = [var, len(pre), pre.mean(), len(post), post.mean(), mean_d_perc, stat, pval, sig]
            res.append(row)

# Turn results list into dataframe for presentation
results_ttest = pd.DataFrame(res, columns=['Variable', 'n Pre-VAR', 'Pre-VAR mean', 'n Post-VAR', 'Post-VAR mean',
                                           'Difference %', 'T-statistic', 'P-value', 'Significance'])

# Save result in csv format and present here
results_ttest.round(decimals=3).to_csv(f"{filedir}/VAR_Analysis_Ttests_{today}.csv", index=False)
results_ttest.round(decimals=3)
