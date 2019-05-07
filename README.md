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

Exact usage on transplant data:  
`python step7_to_long_format.py -i /home/sakre/Data/serial_transplant_data/step\ 7/ -o /home/sakre/Data/serial_transplant_data/step7_long/ -p M`

### Consolidating reformatted data to one file
  1. In the root directory of this repository run:  
  `python consolidate_data.py -i \[LONG_FORMAT_INPUT_DIRECTORY_PATH\] -o \[OUTPUT_DIRECTORY_PATH\]`  
Example:  
`python consolidate_data.py -i long_format_outputs`  

Exact usage on transplant data:  
`python consolidate_data.py -i /home/sakre/Data/serial_transplant_data/step7_long -o /home/sakre/Data/serial_transplant_data -g /home/sakre/Data/serial_transplant_data/mouse_id_group.csv`

- Output directory optional, defaults to current directory (generally repository root)
- This step names the output file the same as the first input file it finds, but replaces M### with
  M_all in the file name

### Calculating Lineage Bias
  1. In the root directory of the repository run:
  `python lineage_bias.py -i [PATH_TO_CONSOLIDATED_LONG_FORMAT_DATA] -o [PATH_TO_OUTPUT_FOLDER] -c [PATH_TO_FACS_CELL_COUNT_FILE]`  

Example from serial transplantation data:  
`python lineage_bias.py -i ~/Data/serial_transplant_data/M_allAniaAnia\ serial\ transpl_percent-engraftment_121018_long.csv -c ~/Data/serial_transplant_data/WBC\ serial\ transpl\ 111618.txt -o ~/Data/serial_transplant_data/lineage_bias`

### Calculate 'Rest of Clones' Data
1. In the root directory run:
```
python rest_of_clones_calc.py \
    --gfp [GFP Percentage FACS File] \
    --wbcs [Cell Count FACS File] \
    --donor [Donor Chimerism File] \
    --group [CSV with columns mouse_id and group, labels mice with phenotypic group] \
    --timepoint-col ['day', 'month', or 'gen'] \
    --baseline-timepoint [Timepoint to use as reference for lineage bias] \
    -o [Directory to store 'rest of clones' data]
```

#### Example
```

GFP_FILE=~/Data/stemcell_aging/OT_2.0_GFP_051818.txt
DONOR_FILE=~/Data/stemcell_aging/OT_2.0_donor_051818.txt
WBC_FILE=~/Data/stemcell_aging/OT_2.0_WBCs_051818-modified.txt
GROUP_FILE=~/Data/stemcell_aging/mouse_id_group.csv
OUTPUT_DIR=~/Data/stemcell_aging/rest_of_clones
TIMEPOINT_COL='month'
BASELINE_TIMEPOINT='4'


python rest_of_clones_calc.py \
    --gfp $GFP_FILE \
    --wbcs $WBC_FILE \
    --donor $DONOR_FILE \
    --group $GROUP_FILE \
    --timepoint-col $TIMEPOINT_COL \
    --baseline-timepoint $BASELINE_TIMEPOINT \
    -o $OUTPUT_DIR
```

### Plotting Data
  1. In the root directory of the repository run:  
```
  python plot_data.py \
      -i [Long Formatted & Consolidated Step 7 Output] \
      -o [Desired Output Folder for Saved Graphs] \
      -l [Lineage Bias Data File] \
      -c [FACS Cell Count Data] \
      -r [Path to Folder Containing Rest of Clones Calculations] \
      -g [Desired Graph Type] 
```
#### Additional Optional Arguments
```
      -t [Threshold to filter certain graphs by] \
      -a [Cumulative Abundance To Calculate Abundance Threshold] \
      -b [Lineage Bias Cutoff for Filtering] \
      -y [Column to Plot on Y-Axis for certain Graphs] \
      -f [Minimum abundance either Gr or B must pass to be used in Lineage Bias] \
      --time-change [Type of Time Change, 'across' or 'between'] \
      -p [Wildcard type option, different for different graphs] \
      --by-gen [Set if plotting serial transplant data] \
      -s [Set to save graphs] \
      --limit-gen [Set to only plot gen 1-3 of serial transplant data] \
      --by-mouse [Set to plot each mouse seperately for certain graphs] \
      --by-clone [Set to plot individual clones, colored by mouse_id]
```

#### Example:
  ```
  Y_COL="NA"
  ABUNDANCE_CUTOFF=0
  BIAS_CUTOFF=0.5
  THRESHOLD=0
  GRAPH_TYPE="contrib_change_cell"
  LIN_BIAS_MIN=0.05
  TIME_CHANGE="NA"
  OPTION="1"
  python plot_data.py -i ~/Data/serial_transplant_data/M_allAniaAnia\ serial\ transpl_percent-engraftment_121018_long.csv \
      -o ~/Data/serial_transplant_data/Graphs \
      -l ~/Data/serial_transplant_data/lineage_bias/lineage_bias_t0-0_from-counts.csv \
      -c ~/Data/serial_transplant_data/'WBC serial transpl 121018.txt' \
      --bias-change ~/Data/serial_transplant_data/lineage_bias/bias_change_t0-0_from-counts.csv \
      -r ~/Data/serial_transplant_data/rest_of_clones \
      -g $GRAPH_TYPE \
      -t $THRESHOLD \
      -a $ABUNDANCE_CUTOFF \
      -b $BIAS_CUTOFF \
      -y $Y_COL \
      -f $LIN_BIAS_MIN \
      --time-change $TIME_CHANGE \
      -p $OPTION \
      --by-gen -s --limit-gen
  ```


### Testing
In the root directory of the repository run:  
`pytest`

  

---
By Samir Akre
 