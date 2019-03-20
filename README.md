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
  - __requirements.txt:__ File of dependencies used to generate  python3 environment able to run scripts


## Usage Instructions
### Setting up environment
  1. Load virtual environment manager of choice using python 3.7.x
      - Anaconda, virtualenv, etc.
  2. In the environment run:  
  `pip install -r requirements.txt`

### Reformatting step7 output
  1. Create new output directory  
  `mkdir step7_outut_reformat` 
  2. Place all step7 outputs to be analyzed in to a single folder
  3. In the root directory of this repository run:  
  `python step7_to_long_format.py -i \[INPUT_DIRECTORY_PATH\] -o \[OUTPUT_DIRECTORY_PATH\]`

Example:  
`mkdir long_format_outputs`  
`python step7_to_long_format.py -i ./inputs -o ./long_format_outputs`

### Consolidating reformatted data to one file
  1. In the root directory of this repository run:  
  `python consolidate_data.py -i \[LONG_FORMAT_INPUT_DIRECTORY_PATH\] -o \[OUTPUT_DIRECTORY_PATH\]`  
Example:  
`python consolidate_data.py -i long_format_outputs`  

- Output directory optional, defaults to current directory (generally repository root)
- This step names the output file the same as the first input file it finds, but replaces M### with
  M_all in the file name

### Calculating Lineage Bias
  2. In the root directory of the repository run:
  `python lineage_bias.py`  
  - Outputs file `lineage_bias.csv` in root directory of repository.
  - Currently requires manually changing variable in main() function to change path of input file
  - TODO: Add argument variables for output directory/filename and input file

### Plotting Data
  1. Open plot_data.py
  2. Edit main() function to change input file to desired file to analyze 
  3. Call different plotting functions from above as desired

  - TODO: Refactor to just be a command line called script


### Testing
In the root directory of the repository run:  
`pytest`

  

---
By Samir Akre
 