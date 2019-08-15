### Process Step 7 Output Data, USE AS TEMPLATE

## Variable Assignments
ROOT_DATA_DIR=

# Results from the 7 step processing pipeline
STEP_7_RESULTS_DIR=$ROOT_DATA_DIR/step_7

# Prefix to filter for when scanning step7 data file names
# IMPORTANT -- CODE MAY FAIL IF PREFIX INCORRECT
PREFIX=M

# Desired output folder for long formatted data, must exist before running script
LONG_FORMAT_OUTPUT_DIR=$ROOT_DATA_DIR/step_7_long

# Desired output location for consolidated long format data ( will be one file), must exist before running script
CONSOLIDATED_OUTPUT_DIR=$ROOT_DATA_DIR

# Manually generated csv file with columns 'mouse_id', 'group'
MOUSE_GROUP_PHENOTYPE_FILE=

# Desired output for lineage bias data, must exist before running script
LINEAGE_BIAS_DIR=$ROOT_DATA_DIR/lineage_bias

# GFP Percentage file
GFP_FILE=$ROOT_DATA_DIR/GFP*.txt

# Donor Chimerism Percentage file
DONOR_FILE=$ROOT_DATA_DIR/donor*.txt

# White Blood Cell FACS count file for normalizing lineage bias
WBC_FILE=$ROOT_DATA_DIR/WBC*.txt


# Column to look for time in, use "month" for time series data, and "gen" for serial transplantation
TIMEPOINT_COL='day'

# Time point to use as 'balanced' for lineage bias analysis. Use 4 for time series, and 1 for serial transplant
BASELINE_TIMEPOINT='by_mouse'

# Output directory for 'rest of clones' analysis, must exist before running this
REST_OF_CLONES_OUTPUT_DIR=$ROOT_DATA_DIR/rest_of_clones


# Representative cell type to use for lineage bias calculation
LYMPHOID_CELL_TYPE='b'
MYELOID_CELL_TYPE='gr'


# Create directories if they do not exist
mkdir -p $LONG_FORMAT_OUTPUT_DIR $CONSOLIDATED_OUTPUT_DIR $LINEAGE_BIAS_DIR $REST_OF_CLONES_OUTPUT_DIR

## Run Processing Scripts

# Format the output files of step 7 ( 1 file per mouse ) to long format (1 file per mouse)
python step7_to_long_format.py \
    -i $STEP_7_RESULTS_DIR \
    -p $PREFIX \
    -o $LONG_FORMAT_OUTPUT_DIR

# Consolidate long format data in to one file (1 file per experiment)
python consolidate_data.py \
    -i $LONG_FORMAT_OUTPUT_DIR \
    -o $CONSOLIDATED_OUTPUT_DIR \
    -g $MOUSE_GROUP_PHENOTYPE_FILE

# Calculate lineage bias, save as csv
python lineage_bias.py \
    -i $CONSOLIDATED_OUTPUT_DIR/${PREFIX}*_long.csv\
    -c $WBC_FILE \
    -o $LINEAGE_BIAS_DIR \
    --timepoint-col $TIMEPOINT_COL \
    --baseline-timepoint $BASELINE_TIMEPOINT \
    --lymph $LYMPHOID_CELL_TYPE \
    --myel $MYELOID_CELL_TYPE


# Calculate lineage bias and abundance for Untracked clones and host clones by treating them as indibidual clones
python rest_of_clones_calc.py \
    --gfp $GFP_FILE \
    --wbcs $WBC_FILE \
    --donor $DONOR_FILE \
    --group $MOUSE_GROUP_PHENOTYPE_FILE \
    --timepoint-col $TIMEPOINT_COL \
    --baseline-timepoint $BASELINE_TIMEPOINT \
    -o $REST_OF_CLONES_OUTPUT_DIR \
    --lymph $LYMPHOID_CELL_TYPE \
    --myel $MYELOID_CELL_TYPE