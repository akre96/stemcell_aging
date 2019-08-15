
# OPTIONS FOR CERTAIN VARIABLES
Y_COL="myeloid_percent_engraftment"
Y_COL="lymphoid_percent_engraftment"
Y_COL="lineage_bias"
GROUP='aging_phenotype'
GROUP='no_change'
GROUP='all'
CHANGE_TYPE='Changed'
CHANGE_TYPE='Unchanged'

# SELEcT PROJECT AND WHERE CODE LIVES
root_data_folder=[PATH TO PROJECT]
root_code_folder=[PATH TO CODE]
cd $root_code_folder


# Default Parameters
Y_COL="None"
GROUP='all'
ABUNDANCE_CUTOFF=0
BIAS_CUTOFF=0
THRESHOLD=0
GRAPH_TYPE="hsc_abund_bias_last"
LIN_BIAS_MIN=0.05
TIME_CHANGE="NA"
TIMEPOINT="None"
OPTION="default"

# Run Plotting function
python plot_data.py -i ${root_data_folder}/*_long.csv \
    -o ${root_data_folder}/Graphs \
    -l ${root_data_folder}/lineage_bias/lineage_bias.csv \
    -c ${root_data_folder}/WBC*.txt \
    --gfp ${root_data_folder}/GFP*.txt \
    --donor ${root_data_folder}/donor*.txt \
    -r ${root_data_folder}/rest_of_clones \
    -g $GRAPH_TYPE \
    -t $THRESHOLD \
    -a $ABUNDANCE_CUTOFF \
    -b $BIAS_CUTOFF \
    -y $Y_COL \
    -f $LIN_BIAS_MIN \
    --group $GROUP \
    --time-change $TIME_CHANGE \
    --timepoint $TIMEPOINT \
    -p $OPTION \
#    --by-gen \
#    --by-group \
#    --by-count \
#    --by-clone \
#    --by-sum \
#    --by-average \
#    --timepoint \
#    --change-type $CHANGE_TYPE \
#    -s

  
