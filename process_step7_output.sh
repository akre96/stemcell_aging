## Process Step 7 Output Data

# Results from the 7 step processing pipeline
STEP_7_RESULTS_DIR=~/Data/stemcell_aging/aging_proc_03122019

# Desired output for long formatted data
OUTPUT_DIR=~/Data/stemcell_aging/aging_proc_03122019_long

# Desired output location for consolidated long format data ( will be one file)
CONSOLIDATED_OUTPUT_DIR=~/Data/stemcell_aging/aging_proc_03122019_long_consolidated

# Manually generated csv file with columns 'mouse_id', 'group'
MOUSE_GROUP_PHENOTYPE_FILE=~/Data/stemcell_aging/mouse_id_group_aging.csv

# Desired output for lineage bias data
LINEAGE_BIAS_DIR=~/Data/stemcell_aging/lineage_bias

# White Blood Cell count file for normalizing lineage bias
WBC_COUNTS_FILE=~/Data/stemcell_aging/OT_2.0_WBCs_051818-modified.txt

# Prefix to filter for when scanning step7 data file names
PREFIX=Ania_M


python step7_to_long_format.py \
    -i $STEP_7_RESULTS_DIR \
    -p $PREFIX \
    -o $OUTPUT_DIR

python consolidate_data.py \
    -i $OUTPUT_DIR \
    -o $CONSOLIDATED_OUTPUT_DIR \
    -g $MOUSE_GROUP_PHENOTYPE_FILE

python lineage_bias.py \
    -i $CONSOLIDATED_OUTPUT_DIR/${PREFIX}_all*_long.csv\
    -c $WBC_COUNTS_FILE \
    -o $LINEAGE_BIAS_DIR \