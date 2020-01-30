from typing import List, Any
from itertools import combinations
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from colorama import init, Fore, Back, Style

init(autoreset=True)

def replace_underscore_dot(string: Any):
    return str(string).replace('_', ' ').replace('.', '-').title()

def print_p_value(context: str, p_value: float, mean: float = None, show_ns: bool = False):
    """ Print P-Value, styke by significance
    
    Arguments:
        context {str} -- What to print just before P-Value
        p_value {float}
    """
    text = context \
            + ' P-Value: ' + str(p_value)
    if mean is not None:
        text += ' Mean Difference: ' + str(mean)

    if p_value < 0.001:
        print(
            Fore.WHITE + Back.CYAN + Style.BRIGHT + text + ' ***'
        )
    elif p_value < 0.01:
        print(
            Fore.CYAN + Style.BRIGHT + text + ' **'
        )
    elif p_value < 0.05:
        print(
            Fore.CYAN + text + ' *'
        )
    elif show_ns:
        print(
            Fore.WHITE + text
        )

def ttest_rel_at_each_time(
        data: pd.DataFrame,
        value_col: str,
        timepoint_col: str,
        group_col: str,
        match_cols: List[str],
        merge_type: str,
        overall_context: str,
        fill_na: Any,
        show_ns: bool,
    ):
    match_col_str = '-'.join(match_cols)
    times = data[timepoint_col].unique()
    print(
        Fore.CYAN + Style.BRIGHT
        + '\nPerforming Paired T-Test on '
        + overall_context
        + ' between ' + group_col.title() + 's at each ' + replace_underscore_dot(timepoint_col)
    )
    groups = data[group_col].unique()
    for time, t_df in data.groupby(timepoint_col):
        for (g1, g2) in combinations(groups, 2):
            g1_df = t_df[t_df[group_col] == g1]
            g2_df = t_df[t_df[group_col] == g2]
            merged = g1_df.merge(
                g2_df,
                on=match_cols,
                how=merge_type,
                validate='1:1',
                suffixes=['_1', '_2']
            )
            if fill_na is not None:
                merged = merged.fillna(
                    value=fill_na,
                )

            count = len(merged)
            if count <= 1:
                p_value = 1
            else:
                _, p_value = stats.ttest_rel(
                    merged[value_col + '_1'],
                    merged[value_col + '_2'],
                )
            
            context: str = timepoint_col.title() + ' ' + replace_underscore_dot(time) + ' ' \
                + replace_underscore_dot(group_col) + ' ' + str(g1) \
                + ' vs ' + str(g2) \
                + ', n ' + match_col_str + ': ' + str(count)
            print_p_value(context, p_value, show_ns=show_ns)


def ttest_1samp(
        data: pd.DataFrame,
        value_var: str,
        group_vars: List[str],
        null_mean: float,
        overall_context: str,
        show_ns: bool,
        handle_nan: str,
        handle_inf: str,
    ):
    print(
        Fore.CYAN + Style.BRIGHT
        + '\nPerforming 1 Sample T-Test on '
        + overall_context
        + ' difference from ' + str(null_mean)
        + ' with Bonferroni adjustment at 0.05'
    )
    if handle_inf == 'omit':
        len_inf = len(data[
            (data[value_var] == np.inf) |\
            (data[value_var] == -np.inf)
            ])
        data = data[data[value_var] != np.inf]
        data = data[data[value_var] != -np.inf]
        print (Fore.YELLOW + ' Ignoring Infinite Values T-Test, n = ' + str(len_inf))

    if handle_nan == 'omit':
        len_nan = len(data[data[value_var].isnull()])
        print (Fore.YELLOW + ' Ignoring NaN Values T-Test, n = ' + str(len_nan))
    
    raw_p_vals = []
    comparisons = []
    means = []
    for group_var, g_df in data.groupby(group_vars):
        _, p_value = stats.ttest_1samp(
            g_df[value_var],
            popmean=null_mean,
            nan_policy=handle_nan,
        )
        g_len = len(g_df)
        mean_val = g_df[value_var].mean()
        context: str = '\t' + ' '.join(group_var).title() \
            + ', Mean: ' + str(mean_val) + ', Count: ' + str(g_len) \
            + ', Mice: ' + str(g_df.mouse_id.nunique())
        raw_p_vals.append(p_value)
        comparisons.append(context)
        means.append(mean_val)
    
    corr_vals = multipletests(
        raw_p_vals,
        alpha=0.05,
        method='bonferroni'
        )[1]
    comp_p = zip(comparisons, corr_vals, means)
    for (comp, p, mean) in comp_p:
        print_p_value(
            comp,
            p,
            mean=mean,
            show_ns=show_ns

        )

def ind_ttest_between_groups_at_each_time(
        data: pd.DataFrame,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
        group_col: str = 'group',
    ):
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test on ' 
        + overall_context
        + ' between ' + group_col.title() + 's at each ' + replace_underscore_dot(timepoint_col)
    )

    groups = data[group_col].unique()
    raw_p_vals = []
    comparisons = []
    means = []
    for time, t_df in data.groupby(timepoint_col):
        for g1, g2 in combinations(groups, 2):
            g1_df = t_df[t_df[group_col] == g1]
            g2_df = t_df[t_df[group_col] == g2]
            _, p_value = stats.ttest_ind(
                g1_df[test_col],
                g2_df[test_col],
            )
            n0 = len(g1_df)
            n1 = len(g2_df)
            mean_diff = g1_df[test_col].mean() \
                - g2_df[test_col].mean()
            context: str = '\t' + timepoint_col.title() + ' ' + str(time) + ' '\
                + g1 + ': ' + str(n0) \
                + ', ' + g2 + ': ' + str(n1)
            raw_p_vals.append(p_value)
            comparisons.append(context)
            means.append(mean_diff)
    
    corr_vals = multipletests(
        raw_p_vals,
        alpha=0.05,
        method='bonferroni'
        )[1]
    comp_p = zip(comparisons, corr_vals, means)
    for (comp, p, mean) in comp_p:
        print_p_value(
            comp,
            p,
            mean=mean,
            show_ns=show_ns

        )

def ind_ttest_group_time(
        data: pd.DataFrame,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
        group_col: str = 'group',
    ):
    times = data[timepoint_col].unique()
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test on ' 
        + overall_context
        + ' between groups at each ' + replace_underscore_dot(timepoint_col)
    )
    groups = data[group_col].unique()
    if len(groups) != 2:
        raise ValueError('Too many groups: ' + ', '.join(groups))
    for t1, t_df in data.groupby(timepoint_col):
        g1_df = t_df[t_df[group_col] == groups[0]]
        g2_df = t_df[t_df[group_col] == groups[1]]
        
        n1 = g1_df.mouse_id.nunique()
        n2 = g2_df.mouse_id.nunique()

        _, p_value = stats.ttest_ind(
            g1_df[test_col],
            g2_df[test_col], 
        )
        context = replace_underscore_dot(t1) + ' ' \
            + replace_underscore_dot(group_col) + ' '\
            + str(groups[0]) + ' Mice: ' + str(n1)\
            + ' vs '\
            + str(groups[1]) + ' Mice: ' + str(n2)
        print_p_value(context, p_value, show_ns=show_ns)



def rel_ttest_group_time(
        data: pd.DataFrame,
        match_cols:List[str],
        merge_type: str,
        fill_na: Any,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
    ):
    times = data[timepoint_col].unique()
    match_col_str = '-'.join(match_cols)
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Paired T-Test on ' 
        + overall_context
        + ' per group between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby('group'):
        for (t1, t2) in combinations(times, 2):
            t1_df = g_df[g_df[timepoint_col] == t1]
            t2_df = g_df[g_df[timepoint_col] == t2]
            merged = t1_df.merge(
                t2_df,
                on=match_cols,
                how=merge_type,
                validate='1:1',
                suffixes=['_1', '_2']
            )
            if fill_na is not None:
                merged = merged.fillna(
                    value=fill_na,
                )

            count = len(merged)
            if count <= 1:
                p_value = 1
            else:
                stat, p_value = stats.ttest_rel(
                    merged[test_col + '_1'],
                    merged[test_col + '_2'],
                )
            
            context: str = replace_underscore_dot(group) + ' ' \
                + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
                + ' vs ' + str(t2) \
                + ', n ' + match_col_str + ': ' + str(count)
            print_p_value(context, p_value, show_ns=show_ns)

    group = 'all'
    for (t1, t2) in combinations(times, 2):
        t1_df = data[data[timepoint_col] == t1]
        t2_df = data[data[timepoint_col] == t2]

        merged = t1_df.merge(
            t2_df,
            on=match_cols,
            how='inner',
            validate='1:1',
            suffixes=['_1', '_2']
        )
        merged = t1_df.merge(
            t2_df,
            on=match_cols,
            how=merge_type,
            validate='1:1',
            suffixes=['_1', '_2']
        )
        if fill_na is not None:
            merged = merged.fillna(
                value=fill_na,
            )
        count = len(merged)
        if count <= 1:
            p_value = 1
        else:
            stat, p_value = stats.ttest_rel(
                merged[test_col + '_1'],
                merged[test_col + '_2'],
            )
        
        context = replace_underscore_dot(group) + ' ' \
            + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
            + ' vs ' + str(t2) \
            + ', n ' + match_col_str + ': ' + str(count)
        print_p_value(context, p_value, show_ns=show_ns)


def signed_ranksum_test_group_time(
        data: pd.DataFrame,
        match_cols: List[str],
        merge_type: str,
        fill_na: Any,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
        group_col: str = 'group',
    ):
    times = data[timepoint_col].unique()
    match_col_str = '-'.join(match_cols)
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Wilcoxon Signed Rank on ' 
        + overall_context
        + ' per ' + group_col + ' between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby(group_col):
        for (t1, t2) in combinations(times, 2):
            t1_df = g_df[g_df[timepoint_col] == t1]
            t2_df = g_df[g_df[timepoint_col] == t2]
            merged = t1_df.merge(
                t2_df,
                on=match_cols,
                how=merge_type,
                validate='1:1',
                suffixes=['_1', '_2']
            )
            if fill_na is not None:
                merged = merged.fillna(
                    value=fill_na,
                )

            count = len(merged)
            if count <= 1:
                p_value = 1
            else:
                stat, p_value = stats.wilcoxon(
                    merged[test_col + '_1'],
                    merged[test_col + '_2'],
                )
            
            context: str = replace_underscore_dot(group) + ' ' \
                + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
                + ' vs ' + str(t2) \
                + ', n ' + match_col_str + ': ' + str(count)
            print_p_value(context, p_value, show_ns=show_ns)

    group = 'all'
    for (t1, t2) in combinations(times, 2):
        t1_df = data[data[timepoint_col] == t1]
        t2_df = data[data[timepoint_col] == t2]

        merged = t1_df.merge(
            t2_df,
            on=match_cols,
            how='inner',
            validate='1:1',
            suffixes=['_1', '_2']
        )
        merged = t1_df.merge(
            t2_df,
            on=match_cols,
            how=merge_type,
            validate='1:1',
            suffixes=['_1', '_2']
        )
        if fill_na is not None:
            merged = merged.fillna(
                value=fill_na,
            )
        count = len(merged)
        if count <= 1:
            p_value = 1
        else:
            stat, p_value = stats.wilcoxon(
                merged[test_col + '_1'],
                merged[test_col + '_2'],
            )
        
        context = replace_underscore_dot(group) + ' ' \
            + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
            + ' vs ' + str(t2) \
            + ', n ' + match_col_str + ': ' + str(count)
        print_p_value(context, p_value, show_ns=show_ns)

def ranksums_test_group(
        data: pd.DataFrame,
        test_col: str,
        overall_context: str,
        show_ns: bool,
        group_col: str = 'group'
):
    groups = data[group_col].unique()

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Bonferonni Corrected Ranksums Test on ' 
        + overall_context
        + ' between ' + group_col 
    )
    raw_p_vals = []
    comparisons = []
    means = []
    for g1, g2 in combinations(groups, 2):
        g1_df = data[data[group_col] == g1]
        g2_df = data[data[group_col] == g2]
        _, p_value = stats.ranksums(
            g1_df[test_col],
            g2_df[test_col],
        )
        n0 = len(g1_df)
        n1 = len(g2_df)
        mean_diff = g1_df[test_col].mean() \
            - g2_df[test_col].mean()
        context: str = '\t'\
            + g1 + ': ' + str(n0) \
            + ', ' + g2 + ': ' + str(n1)
        raw_p_vals.append(p_value)
        comparisons.append(context)
        means.append(mean_diff)
    
    corr_vals = multipletests(
        raw_p_vals,
        alpha=0.05,
        method='bonferroni'
        )[1]
    comp_p = zip(comparisons, corr_vals, means)
    for (comp, p, mean) in comp_p:
        print_p_value(
            comp,
            p,
            mean=mean,
            show_ns=show_ns

        )
def ranksums_test_group_time(
        data: pd.DataFrame,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
        group_col: str = 'group'
):
    times = data[timepoint_col].unique()
    groups = data[group_col].unique()
    if len(groups) != 2:
        print(data.groupby(group_col).code.nunique())
        print(groups)
        if 'aging_phenotype' in groups:
            groups = ['no_change', 'aging_phenotype']
        else:
            raise ValueError('Too many groups: ' + ', '.join([str(x) for x in groups]))

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Bonferonni Corrected Ranksums Test on ' 
        + overall_context
        + ' between ' + group_col + ' at each ' + replace_underscore_dot(timepoint_col)
    )
    raw_p_vals = []
    comparisons = []
    means = []
    for t1, t_df in data.groupby(timepoint_col):
        _, p_value = stats.ranksums(
            t_df[t_df[group_col] == groups[0]][test_col],
            t_df[t_df[group_col] == groups[1]][test_col],
        )
        n0 = len(t_df[t_df[group_col] == groups[0]][test_col])
        n1 = len(t_df[t_df[group_col] == groups[1]][test_col])
        mean_diff = t_df[t_df[group_col] == groups[0]][test_col].mean() \
            - t_df[t_df[group_col] == groups[1]][test_col].mean()
        context: str = '\t' + timepoint_col.title() + ' ' + str(t1) \
            + ', ' + groups[0] + ': ' + str(n0) \
            + ', ' + groups[1] + ': ' + str(n1)
        raw_p_vals.append(p_value)
        comparisons.append(context)
        means.append(mean_diff)
    
    corr_vals = multipletests(
        raw_p_vals,
        alpha=0.05,
        method='bonferroni'
        )[1]
    comp_p = zip(comparisons, corr_vals, means)
    for (comp, p, mean) in comp_p:
        print_p_value(
            comp,
            p,
            mean=mean,
            show_ns=show_ns

        )


def one_way_ANOVArm(
    data: pd.DataFrame,
    timepoint_col: str,
    id_col: str,
    value_col: str,
    overall_context: str,
    show_ns: bool,
    match_cols: List[str],
    merge_type: str,
    fill_na: Any,
) -> None:
    if fill_na is not None:
        index_cols = list(set([id_col] + match_cols))
        times = data[timepoint_col].unique()
        data = data.pivot_table(
            values=value_col,
            index=index_cols,
            columns=timepoint_col,
            fill_value=fill_na,
        )
        data = data.reset_index().melt(
            id_vars=index_cols,
            value_vars=times,
            value_name=value_col
        )

    model = AnovaRM(data, value_col, id_col, within=[timepoint_col])
    res = model.fit()
    p_value = res.anova_table['Pr > F'].tolist()
    if len(p_value) > 1:
        raise ValueError('Too many p values found: ' + str(p_value))

        print(
            Fore.CYAN + Style.BRIGHT
            + '\nPerforming One Way Repeated Measurement ANOVA and bonferroni adjusted Paired T-Test for ' + overall_context.title()
        )
    print_p_value(
        context='ANOVA ' + overall_context,
        p_value=p_value[0],
        show_ns=show_ns,
    )
    times = data[timepoint_col].unique()
    if (p_value[0] < 0.05) or show_ns:
        raw_p_vals = []
        comparisons = []
        for (t1, t2) in combinations(times, 2):
            t1_df = data[data[timepoint_col] == t1]
            t2_df = data[data[timepoint_col] == t2]
            merged = t1_df.merge(
                t2_df,
                on=match_cols,
                how=merge_type,
                validate='1:1',
                suffixes=['_1', '_2']
            )
            if fill_na is not None:
                merged = merged.fillna(
                    value=fill_na,
                )

            count = len(merged)
            if count <= 1:
                p_value = 1
            else:
                stat, p_value = stats.ttest_rel(
                    merged[value_col + '_1'],
                    merged[value_col + '_2'],
                )
            raw_p_vals.append(p_value)
            n_counts = (
                len(t1_df[match_cols].drop_duplicates()),
                len(t2_df[match_cols].drop_duplicates()),
                )
            comp = str(t1) + '(n=' + str(n_counts[0]) + ')' +\
                ' vs ' + str(t2) + '(n=' + str(n_counts[1]) + ')'
            comparisons.append(comp)

        corr_vals = multipletests(
            raw_p_vals,
            alpha=0.05,
            method='bonferroni'
            )[1]
        comp_p = zip(comparisons, corr_vals)
        for (comp, p) in comp_p:
            print_p_value(
                '\t' + overall_context + ' ' + comp,
                p,
                show_ns=show_ns

            )

def friedman_wilcoxonSignedRank(
    data: pd.DataFrame,
    timepoint_col: str,
    id_col: str,
    value_col: str,
    overall_context: str,
    show_ns: bool,
    match_cols: List[str],
    merge_type: str,
    fill_na: Any,
    aggfunc: Any = np.mean,
) -> None:
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Friedman Chi-squared and bonferroni adjusted Wilcoxon Signed Rank Test for ' + overall_context.title()
    )
    measurements = data.pivot_table(
        values=value_col,
        index=id_col,
        columns=timepoint_col,
        aggfunc=aggfunc,
        fill_value=fill_na
    )
    measure_array = [measurements[col] for col in measurements.columns]
    p_value = stats.friedmanchisquare(*measure_array)[1]
    print_p_value(
        context='Friedman ' + overall_context,
        p_value=p_value,
        show_ns=show_ns,
    )
    times = data[timepoint_col].unique()
    if (p_value < 0.05) or show_ns:
        raw_p_vals = []
        comparisons = []
        for (t1, t2) in combinations(times, 2):
            t1_df = data[data[timepoint_col] == t1]
            t2_df = data[data[timepoint_col] == t2]
            merged = t1_df.merge(
                t2_df,
                on=match_cols,
                how=merge_type,
                validate='1:1',
                suffixes=['_1', '_2']
            )
            if fill_na is not None:
                merged = merged.fillna(
                    value=fill_na,
                )

            count = len(merged)
            if count <= 1:
                p_value = 1
            else:
                _, p_value = stats.wilcoxon(
                    merged[value_col + '_1'],
                    merged[value_col + '_2'],
                )
            raw_p_vals.append(p_value)
            comp = str(t1) +\
                ' vs ' + str(t2) + '(n=' + str(count) + ')'
            comparisons.append(comp)

        corr_vals = multipletests(
            raw_p_vals,
            alpha=0.05,
            method='bonferroni'
            )[1]
        comp_p = zip(comparisons, corr_vals)
        for (comp, p) in comp_p:
            print_p_value(
                '\t' + overall_context + ' ' + comp,
                p,
                show_ns=show_ns

            )

def anova_oneway(
    data: pd.DataFrame,
    category_col: str,
    value_col: str,
    overall_context: str,
    show_ns: bool,
) -> None:
    F, p = stats.f_oneway(
        *[data[data[category_col] == cat][value_col] for cat in data[category_col].unique()]
    )
    print_p_value(
        '\n' + overall_context + ' One Way ANOVA' ,
        p,
        show_ns=show_ns

    )
    if p < 0.05:
        comp = pairwise_tukeyhsd(data[value_col], data[category_col])
        print(comp)
        print(data.groupby(category_col).nunique())
