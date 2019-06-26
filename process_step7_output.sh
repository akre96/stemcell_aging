### Process Step 7 Output Data
## NOTE: Step 7 outpur directory must exist

## Variable Assignments 
ROOT_DATA_DIR=~/Data/stemcell_aging

# Results from the 7 step processing pipeline
STEP_7_RESULTS_DIR=~/Data/stemcell_aging/aging_proc_03122019

# Desired output folder for long formatted data, must exist before running script
OUTPUT_DIR=~/Data/stemcell_aging/aging_proc_03122019_long

# Desired output location for consolidated long format data ( will be one file), must exist before running script
CONSOLIDATED_OUTPUT_DIR=~/Data/stemcell_aging/aging_proc_03122019_long_consolidated

# Manually generated csv file with columns 'mouse_id', 'group'
MOUSE_GROUP_PHENOTYPE_FILE=~/Data/stemcell_aging/mouse_id_group_aging.csv

# Desired output for lineage bias data, must exist before running script
LINEAGE_BIAS_DIR=~/Data/stemcell_aging/lineage_bias

# GFP Percentage file
GFP_FILE=~/Data/stemcell_aging/OT_2.0_GFP_051818.txt

# Donor Chimerism Percentage file
DONOR_FILE=~/Data/stemcell_aging/OT_2.0_donor_051818.txt

# White Blood Cell FACS count file for normalizing lineage bias
WBC_FILE=~/Data/stemcell_aging/OT_2.0_WBCs_051818-modified.txt

# Prefix to filter for when scanning step7 data file names
PREFIX=Ania_M

# Column to look for time in, use "month" for time series data, and "gen" for serial transplantation
TIMEPOINT_COL='month'

# Time point to use as 'balanced' for lineage bias analysis. Use 4 for time series, and 1 for serial transplant
BASELINE_TIMEPOINT='4'

# Output directory for 'rest of clones' analysis, must exist before running this
REST_OF_CLONES_OUTPUT_DIR=~/Data/stemcell_aging/rest_of_clones



## 10x serial transplant Presets
#root_data_dir=~/Data/serial_transplant_10x
#STEP_7_RESULTS_DIR=${root_data_dir}/step7
#OUTPUT_DIR=${root_data_dir}/step7_long
#CONSOLIDATED_OUTPUT_DIR=${root_data_dir}
#MOUSE_GROUP_PHENOTYPE_FILE=${root_data_dir}/mouse_id_group_serial.csv
#LINEAGE_BIAS_DIR=${root_data_dir}/lineage_bias
#GFP_FILE="${root_data_dir}/GFP*.txt"
#DONOR_FILE="${root_data_dir}/DONOR*.txt"
#WBC_FILE="${root_data_dir}/WBC*.txt"
#PREFIX=M
#TIMEPOINT_COL='day'
#BASELINE_TIMEPOINT='127'
#REST_OF_CLONES_OUTPUT_DIR=${root_data_dir}/rest_of_clones

# Marys Data Processing Preset
#root_data_dir=~/Data/mary_project_0-01
#STEP_7_RESULTS_DIR=${root_data_dir}/Step7_0-01_060619
#OUTPUT_DIR=${root_data_dir}/step7_long
#CONSOLIDATED_OUTPUT_DIR=${root_data_dir}
#MOUSE_GROUP_PHENOTYPE_FILE=${root_data_dir}/mouse_id_group_mary.csv
#LINEAGE_BIAS_DIR=${root_data_dir}/lineage_bias
#GFP_FILE="${root_data_dir}/GFP*.txt"
#DONOR_FILE="${root_data_dir}/DONOR*.txt"
#WBC_FILE="${root_data_dir}/WBC*.txt"
#PREFIX=M
#TIMEPOINT_COL='day'
#BASELINE_TIMEPOINT='127'
#REST_OF_CLONES_OUTPUT_DIR=${root_data_dir}/rest_of_clones

# Create directories if they do not exist
mkdir -p $OUTPUT_DIR $CONSOLIDATED_OUTPUT_DIR $LINEAGE_BIAS_DIR $REST_OF_CLONES_OUTPUT_DIR

## Run Processing Scripts

# Format the output files of step 7 ( 1 file per mouse ) to long format (1 file per mouse)
python step7_to_long_format.py \
    -i $STEP_7_RESULTS_DIR \
    -p $PREFIX \
    -o $OUTPUT_DIR

# Consolidate long format data in to one file (1 file per experiment)
python consolidate_data.py \
    -i $OUTPUT_DIR \
    -o $CONSOLIDATED_OUTPUT_DIR \
    -g $MOUSE_GROUP_PHENOTYPE_FILE

# Calculate lineage bias, save as csv
python lineage_bias.py \
    -i $CONSOLIDATED_OUTPUT_DIR/${PREFIX}_all*_long.csv\
    -c $WBC_FILE \
    -o $LINEAGE_BIAS_DIR \


# Calculate lineage bias and abundance for Untracked clones and host clones by treating them as indibidual clones
python rest_of_clones_calc.py \
    --gfp $GFP_FILE \
    --wbcs $WBC_FILE \
    --donor $DONOR_FILE \
    --group $MOUSE_GROUP_PHENOTYPE_FILE \
    --timepoint-col $TIMEPOINT_COL \
    --baseline-timepoint $BASELINE_TIMEPOINT \
    -o $REST_OF_CLONES_OUTPUT_DIR