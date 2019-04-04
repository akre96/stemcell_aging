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
### Plotting Data
  1. In the root directory of the repository run:  
  `python plot_data.py`  
    - By default this will plot whatever the most recently created/worked on graph was 
  2. add a `-s` flag to save plot outputs
  3. add a `-g GRAPH_TYPE` to change graph type, the following GRAPH_TYPEs available  
    -  cluster:            Clustered heatmap of present clone engraftment  
    -  venn:               Venn Diagram of clone existance at timepoint  
    -  clone_count:        Bar charts of clone counts by cell type at different thresholds  
    -  lineage_bias_line:  lineplots of lineage bias over time at different abundance from last timepoint  
    -  engraftment_time:   lineplot/swarmplot of abundance of clones with high values at 4, 12, and 14 months  
    -  counts_at_perc:     line or barplot of clone counts where cell-types are filtered at 90th percentile of abundance  
  4. add a `-i` flag to specify location of long format step7 output
  5. add `-o` flag to specify where to output graphs [NOT IMPLEMENTED YET]
  7. add `-l` flag to specify location of lineage_bias file



### Testing
In the root directory of the repository run:  
`pytest`

  

---
By Samir Akre
 