import os
import time
import pandas as pd
import scipy.stats as stats
import numpy as np
from typing import List, Tuple
import itertools
import multiprocessing as mp



def rna_seq_normalized_matrix_to_long(rnaseq_file_path: str) -> pd.DataFrame:
    """ Returns long form rna-seq data from the normalized matrix format

    Arguments:
        rnaseq_file_path {str} -- path to file
          (for example:'10x_data/HCT1_subset-matrix.csv)

    Returns:
        pd.DataFrame -- Long form expression data. Columns: code, gene_id
          expression
    """
    rna_df = pd.read_csv(os.path.join(rnaseq_file_path))\
        .drop(columns=['n'])\
        .rename(columns={'TBC': 'code'})
      
    gene_cols = [c for c in rna_df.columns if c[0] == 'E']
    rna_df = rna_df[['CBC', 'code'] + gene_cols]
    return rna_df.melt(id_vars=['code', 'CBC'], var_name='gene_id', value_name='expression')

def compare_gene_expression_vs_labels(
        data: pd.DataFrame,
        l0: str,
        l1: str,
    ) -> pd.DataFrame:
    """ Compare gene expression for clones in one label_name vs another
    MannWhitney U one directional test computed on cell expression of each group
    
    Arguments:
        data {pd.DataFrame} -- data for one gene and two labels
    
    Returns:
        pd.DataFrame -- one row of data with gene_id, p_value_greater, p_value_less, fold_change, and log2fc
    """
    labels = data['label_name'].unique()
    if not (l0 and l1):
        raise ValueError('No labels')

    if len(labels) != 2:
        raise ValueError('Expected 2 labels in data, recieved ' + str(len(labels)))

    if data['gene_id'].nunique() != 1:
        raise ValueError('Expected 1 Gene, recieved: ' + str(data['gene_id'].nunique()))
    
    if data[data.label_name.isin([l0, l1])].empty:
        raise ValueError('No data found with labels')

    comparison_str = l0 + ' vs ' + l1

    l0_exp = data[data['label_name'] == l0].expression
    l1_exp = data[data['label_name'] == l1].expression
    l0_mean = l0_exp.mean()
    l1_mean = l1_exp.mean()

    if l0_mean == l1_mean:
        return pd.DataFrame.from_dict(
            {
                'gene_id': data.gene_id.unique(),
                'comparison': [comparison_str],
                'p_value_greater': [1],
                'p_value_less': [1],
                'fold_change': [1],
                'log2fc': [0],
                'count_0': l0_exp.count(),
                'count_1': l1_exp.count(),
                'mean_0': [l0_mean],
                'mean_1': [l1_mean],
                'l0': [l0],
                'l1': [l1],
            }
        )
    _, p_val_greater = stats.mannwhitneyu(l0_exp, l1_exp, alternative='greater')
    _, p_val_less = stats.mannwhitneyu(l0_exp, l1_exp, alternative='less')
    
    fold_change = l0_mean/l1_mean
    log2fc = np.log2(fold_change)

    return pd.DataFrame.from_dict(
        {
            'gene_id': data.gene_id.unique(),
            'comparison': [comparison_str],
            'p_value_greater': [p_val_greater],
            'p_value_less': [p_val_less],
            'fold_change': [fold_change],
            'log2fc': [log2fc],
            'count_0': l0_exp.count(),
            'count_1': l1_exp.count(),
            'mean_0': [l0_mean],
            'mean_1': [l1_mean],
            'l0': [l0],
            'l1': [l1],
        }
    )


def generate_rna_seq_label_comparison_df(
        experiment_data_dir: str,
        scramble_data_dir: str,
        replicates: List[str],
        n_scrambles: List[str],
        labels_df: pd.DataFrame,
        label_comparisons: List[Tuple[str, str]],
        filter_genes: bool = False,
        filter_genes_path: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Calculate differences between labeled HSCs using RNA-Seq Data
    Goes through each file in tenx_data_dir that is specified in 'replicates' and 'experiment_types'.
    Imports RNA-Seq data, maps each barcode to a label, calculates expression differences between cells
    assigned to each label. Stores p_values, fold_changes, and cell counts in 'comparison_df'. RNA-Seq
    expression data of mapped clones in experimental data stored in 'experiment_rna_data_df'. barcodes of
    clones that succesfully map to rna-seq data stored in 'mapped_clones'

    
    Arguments:
        tenx_data_dir {str} -- Directory containing tenx_data with file_names: [replicate]_[experiment_type]_subset-matrix.csv
        replicates {List[str]} -- List of runs to look at
        labels_df {pd.DataFrame} -- Dataframe with columns ['code', 'label_int', 'label_name'],
            if patient sample required must have columns ['code', 'label_int', 'label_name', 'patient_sample']
        label_comparisons {List[Tuple[str, str]]} -- List of labels to compare. Each tuple must be of length two, which each member
            corresponding to a label
    
    Keyword Arguments:
        filt_patient_sample {bool} -- If true looks for patient_sample in label_df (default: {False})
        filter_genes {bool} -- Wether to use a subset of genes or not in analysis (default: {False})
        filter_genes_path {str} -- path to list of genes (.npy file) generated using create_filtered_gene_list.ipynb (default: {None})
    
    Raises:
        ValueError: filter_genes set, but no path to filter_genes_path set
        NameError: filt_patient_sample set but patient_sample not found as column in labels_df
        ValueError: Invalid replicate given
        ValueError: More than  two labels in one of the label_comparisons tuples
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] -- comparison_df, mapped_clones, experiment_rna_data_df
    """

    if filter_genes:
        if filter_genes_path is None:
            raise ValueError('filter_genes_path not set, specify path or set filter_genes to False')
        print('Filtering Genes Using List From:', filter_genes_path)
        filtered_genes = filtered_numpy_gene_list(filter_genes_path)

    start = time.process_time()

    comparison_df = pd.DataFrame() # Holds all comparison data from all replicates-comparisons-experiments
    mapped_clones = pd.DataFrame()
    experiment_rna_data_df = pd.DataFrame()
    scrambles = ['scramble-'+str(i) for i in range(n_scrambles)]
    experiment_types = ['experiment'] + scrambles
    # For each replicate and experiment calculate P_value and fold change for genes
    for (replicate, experiment_type) in itertools.product(replicates, experiment_types):
        print('Working on:', replicate, experiment_type, '...')
        
        # RNA-Seq data file path
        if experiment_type == 'experiment':
            file_name = os.path.join(
              experiment_data_dir,
              replicate + '_bridged.csv'
            )
        else:
            file_name = os.path.join(
              scramble_data_dir,
              replicate + '_' + experiment_type + '.csv'
            )
        
        # Import rna_seq data in long format
        print('\t...Importing Data')
        rna_df = rna_seq_normalized_matrix_to_long(file_name)
        if filter_genes:
            print('\t\t Before filtering genes:', len(rna_df))
            rna_df = rna_df.merge(
                filtered_genes,
                how='inner',
                validate='m:1'
            )
            print('\t\t After filtering genes: ', len(rna_df))
          

        # Select only those clones with labels (blood or bone biased)
        labeled_rna_df = rna_df.merge(
            labels_df,
            how='inner',
            validate='m:m'
        )
        if experiment_type == 'experiment':
            labeled_rna_df['replicate'] = replicate
            experiment_rna_data_df = experiment_rna_data_df.append(labeled_rna_df)
            mapped_clones = mapped_clones.append(labeled_rna_df[['code', 'label_name']].drop_duplicates())
        
        for comp in label_comparisons:
            print('\t...Comparing:', comp)
            two_labels_df = labeled_rna_df[labeled_rna_df.label_name.isin(comp)]
            if two_labels_df.label_name.nunique() != 2:
                print('\t\t TWO LABELS NOT FOUND, ONLY SAW:', two_labels_df.label_name.unique())
                continue
            inputs = [
                (g_df, comp[0], comp[1]) for _, g_df in two_labels_df.groupby('gene_id')

            ]

            # Ignores divide by zero and log(inf) errors
            with np.errstate(divide='ignore', invalid='ignore'):

                # Run comparison using all available CPUs
                with mp.Pool() as pool:
                    comp_results = pool.starmap(compare_gene_expression_vs_labels, inputs)
                    comp_df = pd.concat(comp_results) # Dataframe for one experiment-replicate-comparison


            two_labels = two_labels_df['label_name'].unique()
            if len(two_labels) != 2:
                raise ValueError('Expected 2 labels in data, recieved ' + str(len(two_labels)))

            comp_df['replicate'] = replicate
            comp_df['experiment_type'] = experiment_type
            comparison_df = comparison_df.append(comp_df)
        print('Done.')

    print('Time Elapsed:', round(time.process_time() - start), 'seconds')
    return comparison_df.drop_duplicates(), mapped_clones.drop_duplicates(), experiment_rna_data_df

def combine_p_values(
        rows: pd.DataFrame,
        p_value_greater_col: str,
        p_value_less_col: str,
        fold_change_col: str,
        log2fc_col: str,
        **kwargs
    ) -> Tuple[float, float, float, float]:
    """ Use Fishers method to combine p_values for each one-sided test direction
    returns (p_value_less, p_value_greater, fold_change, log2fc)
    """
    p_value_greater = stats.combine_pvalues(rows[p_value_greater_col], method='fisher')[1]
    p_value_less = stats.combine_pvalues(rows[p_value_less_col], method='fisher')[1]
    fold_change = rows[fold_change_col].mean()
    log2fc = rows[log2fc_col].mean()

    return p_value_less, p_value_greater, fold_change, log2fc

def combine_group_p_values(
        comparison_df: pd.DataFrame,
        isolate_group_cols: List[str],
    ) -> pd.DataFrame:
    """ Combines P values given columns to use to isolate
    
    Arguments:
        comparison_df {pd.DataFrame} -- rna-seq gene expression comparison data frame
        isolate_group_cols {List[str]} -- list of columns used to isolate a group
    
    Returns:
        pd.DataFrame -- Dataframe missing isolated groups column with a combined p_value across that group
    """

    g = comparison_df.groupby(isolate_group_cols)

    group_combined = g.apply(
        combine_p_values,
        axis=1,
        return_type='expand',
        p_value_greater_col='p_value_greater',
        p_value_less_col='p_value_less',
        fold_change_col='fold_change',
        log2fc_col='log2fc'
    ).rename("tuple")

    # Series of Tuples is returned by the apply function, code below unpacks tuple in to 4 columns
    tuple_index = group_combined.index
    tuple_list = group_combined.tolist()
    group_combined_df = pd.DataFrame(
        tuple_list,
        tuple_index,
        columns=['p_value_less', 'p_value_greater', 'fold_change', 'log2fc']
    ).reset_index()

    return group_combined_df

def calculate_fpr(
        scramble_df: pd.DataFrame,
        p_vals: List[float],
        p_value_col: str,
        isolate_gene_cols: List[str],
        experiment_data_df: pd.DataFrame
    ):
    """ Calculate False Positive rate for a set of P-values
    
    Arguments:
        scramble_df {pd.DataFrame} -- Grouped data, to isolate all genes from a unique scramble experiment (for example all scramble-0 data from hct2)
        p_vals {List[float]} -- List of p_values to screen through
        p_value_col {str} -- Column to find p_values in g_df
        isolate_gene_cols {List[str]} -- Columns used to isolate one unique scramble set
        experiment_data_df {pd.DataFrame} -- Real/Experiment data (for example hct2 experiment data)
    
    Returns:
        pd.DataFrame -- False positive rates for each p_value,
    """
    fpr_rows = []
    for p_val in p_vals:
        num_exp_genes = experiment_data_df[experiment_data_df[p_value_col] < p_val].gene_id.nunique()
        num_below_p = scramble_df[scramble_df[p_value_col] < p_val].gene_id.nunique()

        # False positive rate defined for the group of genes within a scramble experiment for each comparison made
        # Defined as the ratio of number of genes that pass a p_value threshold in scramble over number that pass in experiment
        if num_exp_genes:
            fpr = num_below_p/num_exp_genes
        else:
            if num_below_p:
                fpr = None
            else:
                fpr = 0

        fpr_row = pd.DataFrame()
        for col in isolate_gene_cols:
            fpr_row[col] = scramble_df[col].unique()
        fpr_row['p_value'] = [p_val]
        fpr_row['fpr'] = [fpr]
        fpr_rows.append(fpr_row)
    return pd.concat(fpr_rows)

def calculate_fpr_genes_vs_comparisons(
        comparison_df: pd.DataFrame,
        isolate_gene_cols: List[str],
        p_value_col: str,
        p_value_cutoff: float,
    ) -> pd.DataFrame:
    """ Calculates false positive rate from rna-seq gene expression comparison_df
    
    Arguments:
        comparison_df {pd.DataFrame} -- Long form data with experiment AND scramble included
        isolate_gene_cols {List[str]} -- columns used to isolate genes in one experiment/scramble-set
        p_value_col {str} -- Column to find p_value in
        p_value_cutoff {float} -- max p_value to calculate fpr in
    
    Returns:
        pd.DataFrame -- experiment comparison dataframe with fpr and p_values
    """
    scramble_data_df = comparison_df[comparison_df.experiment_type != 'experiment']
    experiment_data_df = comparison_df[comparison_df.experiment_type == 'experiment']

    p_vals = experiment_data_df[p_value_col].unique()
    p_vals.sort()
    print('# of P-Values:', len(p_vals))
    print('# of P-Values less than or equal to ', p_value_cutoff, len(p_vals[p_vals <= p_value_cutoff]))
    p_vals = p_vals[p_vals <= p_value_cutoff]

    fpr_rows: List[pd.DataFrame] = []

    # Inputs for parallel computing
    inputs = [
        (
            g_df,
            p_vals,
            p_value_col,
            isolate_gene_cols,
            experiment_data_df,
        )
        for _, g_df in scramble_data_df.groupby(isolate_gene_cols)
    ]

    with mp.Pool() as pool:
        fpr_rows = pool.starmap(calculate_fpr, inputs)


    fpr_df = pd.concat(fpr_rows)
    fpr_df = fpr_df.dropna()
    fpr_df['fpr'] = fpr_df['fpr'].astype(float)
    calc_fpr_cols = [col for col in isolate_gene_cols if col != 'experiment_type'] + ['p_value']

    median_across_scrambles_fpr_df = pd.DataFrame(
        fpr_df.groupby(calc_fpr_cols)['fpr'].median()
    ).reset_index().rename(
        columns={
            'p_value': p_value_col,
            'fpr': p_value_col + '_fpr'
            }
        )


    exp_with_fpr_df = experiment_data_df.merge(
        median_across_scrambles_fpr_df,
        how='outer',
        validate='m:1'
    )
    max_fpr = exp_with_fpr_df[p_value_col + '_fpr'].max()
    exp_with_fpr_df.loc[exp_with_fpr_df[p_value_col + '_fpr'].isna(), p_value_col + '_fpr' ] = max_fpr
    exp_with_fpr_df['nlt_' + p_value_col] = -1 * np.log10(exp_with_fpr_df[p_value_col])
    return exp_with_fpr_df

def filtered_numpy_gene_list(gene_list_path: str) -> pd.DataFrame:
    """ Import numpy saved gene list (expects a .npy file)

    Arguments:
        gene_list_path {str}

    Returns:
        pd.DataFrame -- dataframe with column "gene_id"
    """
    if gene_list_path.split('.')[-1] != 'npy':
        raise ValueError(
            'ERROR: Expected .npy file extension when reading filtered gene list, was given: ' 
            + gene_list_path.split('.')[-1]
        )

    filtered_gene_list = np.load(gene_list_path, allow_pickle=True)
    filtered_gene_list = list(filtered_gene_list.tolist())
    return pd.DataFrame(pd.Series(filtered_gene_list, name='gene_id').str.decode("utf-8"))
