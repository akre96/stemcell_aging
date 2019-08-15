# Stem Cell Aging Analysis tools
__Intended for use in the Rong Lu Lab__
Uses python 3.7.2

## Directory Structure
  - __test:__ Folder for unit testing functions
    - __test_data:__ Folder for curated data with known output
  - __step7_to_long_format.py:__ Takes output of step 7 scripts and formats with columns: code, mouse_id, user, cell_type, day, percent_engraftment
  - __consolidate_data.py:__ takes all long format files specified, and creates single csv output. Used to get all data from all mice in one place. Appends information about mouse group phenotype.
    - current groups are set to either 'no_change' or 'aging_phenotype' based on facs data.
    - aging_phenotype are those mice who's gr population increases and b cell decreases as expected in typical aging
    - no_change are those mice who's gr and b population stay relatively constant
  - __plot_data.py:__ Collection of functions used for plotting formatted data.
  - __lineage_bias.py__: Calculates lineage bias of clones over time based on gr and b cell abundance. Outputs csv file.
  - __statistical_tests.py__: Contains functions for statsitical analysis done commonly throughout projects/plots
  - __parse_facs_data.py:__: Contains functions for ingesting FACS data from the tab deliminated .txt files and transforming in to pandas DataFrame
  - __data_types.py:__: Contains data types used to filter inputs for plot_data.py
  - __time_series_to_STEM.py:__: Function to generate clonal abundance data for use in STEM Clustering program. Primarily just used for 10x + Aging over time project
  - __young_vs_old_HSCS.ipynb__: Jupyter notebook used to compare HSC lineage bias between young and old mice together. 
  - __requirements.txt:__ File of dependencies used to generate  python3 environment able to run scripts
  - __lib/:__ Contains json files explicitly linking mice and other group variables to marker shapes, line styles, and colors
  - __templates/:__ Contains templates for processing projects from step7 output and plotting data


## Usage Instructions
### Setting up environment
  1. Load virtual environment manager of choice using python 3.7.x
      - Anaconda, virtualenv, etc.
      - If you have python3 run `python3 -m venv venv` and then `source venv/bin/activate`
  2. In the environment run:  
  `pip install -r requirements.txt`

## Pre-Processing Steps
  1. Reformatting step7 output
  2. Consolidating reformatted data to one file
  3. Calculate Lineage Bias
  4. Inspect "rest of clone" behavior
    - This corresponds to unbarcoded clones being treated as one barcode

To run all of these steps view the template at `process_step7_output.sh` . This file begins with variable definitions you will need to set in order to select the correct inputs and outputs for data processing.

### Creating the mouse_id to phenotypic group file
  1. In any spreadsheet app create a sheet with two columns with headers: mouse_id, and group respectively.
  2. Fill all mouse_ids in the mouse_id column and corresponding phenotypic group label in the group column
  3. Export this spread sheet as a comma seperated value file (.csv).

### Plotting Data
View file in `templates/plotting_template` for parameters/setup for plotting


### Testing
In the root directory of the repository run:  
`pytest`

  
# Manual Data Cleaning Steps:
- For mouse 3026, Day 122 B cell columns changed from `Ania_M3026D122CB` to `Ania_M3026D122Cb`  
- WBCs, GFP, and DONOR files manually concatenated for aging over time and aging over time 10X mice.




---
By Samir Akre
 