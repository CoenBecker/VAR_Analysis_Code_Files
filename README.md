# VAR_Analysis_Code_Files

**Description**

This Github repository contains the code and files uses for my master thesis. The subject of this thesis is the effect that the Video Assistant Referee (VAR) has on football. 
The focus of the research is the VAR's impact on competitiveness in matches (Uncertainty of Outcome) and competitiveness in the league (Competitive Balance). 

**Files**

This repository contains 4 code files that together produce the results of the research from scratch. 
1. VAR_Analysis_Scrapers.py contains python code that scrapes the desired data from online sources. There is one scraper for match data and another for league rankings of each chosen season.
2. VAR_Analysis_DataPreparation.py contains python code that requires access to the previously compiled data from the 'Scrapers' code and cleans it as well as calculating more variables from it.
3. VAR_Analysis_EDA.py contains python code that executed exploratory data analysis and produces csv/xlsx files containing (selected) descriptive statistics, correlations, histograms, and t-test results from the data files.
4. VAR_Analysis_Regressions.py contains python code that performs various Fixed-Effects and Difference-in-Difference analyses. It also produces one more file containing descriptives and t-test results of specific subsets of the data.

This repository contains 5 data .csv files that the code files produce and/or require to run. 
1. Rankings_All_LeagueSeasons.csv is a file containing all rankings data that the scraper produces, with limited cleaning.
2. Matchdata_main.csv is a file containing all matchdata that the scraper produces, after selecting only a subset of interesting columns to keep.
3. Rankings_Full_Analysis.csv is a file that contains cleaned and prepared data pertaining to league-tables with the full detail on team entries in each table.
4. Rankings_Analysis.csv is a file that contains cleaned and prepared data from the league-tables pertaining to league only, this data is aggregated at the league-level and contains league data but no data on individual teams in the rankings.
5. Matches_Analysis is a file that contains the cleaned and prepared data pertaining to matches.  

**Instructions**

- **Important note**: the VAR_Analysis_Scrapers file sometimes fails to collect the data on rankings due to an apparent irregular bug in line 92, where the code 'pd.read_html(data.text)[0]' fails to retrieve the standings table from the html. In case this happens the code has worked again without any change the next day, but the rest of the files will run in any case and the Rankings_All_LeagueSeasons.csv file can be use to bypass this problematic step and skip ahead in any case.    
- All code files are ready to run and require only basic Python packages such as Pandas, Numpy, BeautifulSoup, Seaborn, Scipy.stats, Statsmodels. However, in some versions of Python and/or depending on code editor the packages 'requests', 'pandas', and 'beautifulsoup4' must first be installed. All three can be installed by running the command 'pip install ' followed by the package name as written here from the command line, which does require that pip itself is installed. 
- All four code files contain code that needs to read and write files to and from a specified directory. Running the code will automatically prompt the setting of this directory, but care must be taken that a correct and safe directory is set.
- Code in the scraper files can be manually adjusted to get a different selection of leagues and seasons from the sources, provided these are available and appropriately specified.
- A word of warning: While the code can in principle be adjusted to different needs and, for example, applied to a different sample and different variables, the proper execution of the (subsequent) code files may be contingent on the preservation of the original parameters. Any changes in code in one place may require adjustment in other places as well.

**File interdependencies instructions**

- To run the VAR_Analysis_Regressions.py file the following data files are required: Matches_Analysis.csv and Rankings_Analysis.csv. 
- To run the VAR_Analysis_EDA.py file the following data files are required: Matches_Analysis.csv, Rankings_Full_Analysis.csv, and Rankings_Analysis.csv.
- To run the VAR_Analysis_DataPreparation.py file the following data files are required: Matchdata_main.csv, Rankings_All_LeagueSeasons.csv.
- The Matchdata_main.csv and Rankings_All_LeagueSeasons.csv files can be downloaded (and placed in the specified directory) from this repository, or be obtained by running the VAR_Analysis_Scrapers.py file.
- The Matches_Analysis.csv, Rankings_Analysis.csv, and Rankings_Full_Analysis.csv files can be downloaded (and placed in the specified directory) from this repository, or be obtained by running the VAR_Analysis_DataPreparation.py file with the required untreated data files.
