from typing import List, Any
from itertools import combinations
import pandas as pd
import numpy as np
import scipy.stats as stats
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
        text += ' Mean: ' + str(mean)

    if p_value < 0.001:
        print(
            Fore.WHITE + Back.CYAN + Style.BRIGHT + text
        )
    elif p_value < 0.01:
        print(
            Fore.CYAN + Style.BRIGHT + text
        )
    elif p_value < 0.05:
        print(
            Fore.CYAN + text
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
    
    for group_var, g_df in data.groupby(group_vars):
        _, p_value = stats.ttest_1samp(
            g_df[value_var],
            popmean=null_mean,
            nan_policy=handle_nan,
        )
        mean_val = g_df[value_var].mean()
        context: str = ' '.join(group_var).title() \
            + ', Mean: ' + str(mean_val)
        print_p_value(context, p_value, show_ns=show_ns)

def ind_ttest_between_groups_at_each_time(
        data: pd.DataFrame,
        test_col: str,
        timepoint_col: str,
        overall_context: str,
        show_ns: bool,
    ):
    times = data[timepoint_col].unique()
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test on ' 
        + overall_context
        + ' between groups at each ' + replace_underscore_dot(timepoint_col)
    )
    for t1, t_df in data.groupby(timepoint_col):
        _, p_value = stats.ttest_ind(
            t_df[t_df.group == 'aging_phenotype'][test_col],
            t_df[t_df.group == 'no_change'][test_col],
        )
        context: str = timepoint_col.title() + ' ' + str(t1) 
        print_p_value(context, p_value, show_ns=show_ns)


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
        _, p_value = stats.ttest_ind(
            t_df[t_df[group_col] == groups[0]][test_col],
            t_df[t_df[group_col] == groups[1]][test_col],
        )
        context: str = timepoint_col.title() + ' ' + str(t1) 
        print_p_value(context, p_value, show_ns=show_ns)

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test on ' 
        + overall_context
        + ' per group between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby(group_col):
        for (t1, t2) in combinations(times, 2):
            t1_df = g_df[g_df[timepoint_col] == t1][test_col]
            t2_df = g_df[g_df[timepoint_col] == t2][test_col]

            stat, p_value = stats.ttest_ind(
                t1_df,
                t2_df, 
            )
            context = replace_underscore_dot(group) + ' ' \
                + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
                + ' vs ' + str(t2)
            print_p_value(context, p_value, show_ns=show_ns)

    group = 'all'
    for (t1, t2) in combinations(times, 2):
        t1_df = data[data[timepoint_col] == t1][test_col]
        t2_df = data[data[timepoint_col] == t2][test_col]

        stat, p_value = stats.ttest_ind(
            t1_df,
            t2_df, 
        )
        context = replace_underscore_dot(group) + ' ' \
            + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
            + ' vs ' + str(t2)
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
        + '\nPerforming Ranksums Test on ' 
        + overall_context
        + ' between ' + group_col + ' at each ' + replace_underscore_dot(timepoint_col)
    )
    for t1, t_df in data.groupby(timepoint_col):
        _, p_value = stats.ranksums(
            t_df[t_df[group_col] == groups[0]][test_col],
            t_df[t_df[group_col] == groups[1]][test_col],
        )
        mean_diff = t_df[t_df[group_col] == groups[0]][test_col].mean() \
            - t_df[t_df[group_col] == groups[1]][test_col].mean()
        context: str = timepoint_col.title() + ' ' + str(t1) 
        print_p_value(context, p_value, mean=mean_diff, show_ns=show_ns)

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Ranksums Test on ' 
        + overall_context
        + ' per ' + group_col + ' between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby(group_col):
        for (t1, t2) in combinations(times, 2):
            t1_df = g_df[g_df[timepoint_col] == t1][test_col]
            t2_df = g_df[g_df[timepoint_col] == t2][test_col]

            _, p_value = stats.ranksums(
                t1_df,
                t2_df,
            )
            mean_diff = t1_df.mean() - t2_df.mean()
            context = replace_underscore_dot(group) + ' ' \
                + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
                + ' vs ' + str(t2)
            print_p_value(context, p_value, mean=mean_diff, show_ns=show_ns)

    group = 'all'
    for (t1, t2) in combinations(times, 2):
        t1_df = data[data[timepoint_col] == t1][test_col]
        t2_df = data[data[timepoint_col] == t2][test_col]

        stat, p_value = stats.ranksums(
            t1_df,
            t2_df, 
        )
        mean_diff = t1_df.mean() - t2_df.mean()
        context = replace_underscore_dot(group) + ' ' \
            + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
            + ' vs ' + str(t2)
        print_p_value(context, p_value, mean=mean_diff, show_ns=show_ns)
