from typing import List, Any
from itertools import combinations
import pandas as pd
import scipy.stats as stats
from colorama import init, Fore, Back, Style

init(autoreset=True)

def replace_underscore_dot(string: str):
    return string.replace('_', ' ').replace('.', '-').title()

def print_p_value(context: str, p_value: float, show_ns: bool = False):
    """ Print P-Value, styke by significance
    
    Arguments:
        context {str} -- What to print just before P-Value
        p_value {float}
    """
    if p_value < 0.001:
        print(
            Fore.WHITE + Back.CYAN + Style.BRIGHT
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif p_value < 0.01:
        print(
            Fore.CYAN + Style.BRIGHT
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif p_value < 0.05:
        print(
            Fore.CYAN 
            + context
            + ' P-Value: ' + str(p_value)
        )
    elif show_ns:
        print(
            Fore.WHITE
            + context
            + ' P-Value: ' + str(p_value)
        )

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

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Independent T-Test on ' 
        + overall_context
        + ' per group between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby('group'):
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
    ):
    times = data[timepoint_col].unique()
    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Rank-Sum T-Test on ' 
        + overall_context
        + ' between groups at each ' + replace_underscore_dot(timepoint_col)
    )
    for t1, t_df in data.groupby(timepoint_col):
        _, p_value = stats.ranksums(
            t_df[t_df.group == 'aging_phenotype'][test_col],
            t_df[t_df.group == 'no_change'][test_col],
        )
        context: str = timepoint_col.title() + ' ' + str(t1) 
        print_p_value(context, p_value, show_ns=show_ns)

    print(
        Fore.CYAN + Style.BRIGHT 
        + '\nPerforming Rank Sum T-Test on ' 
        + overall_context
        + ' per group between ' + replace_underscore_dot(timepoint_col) + 's'
    )
    for group, g_df in data.groupby('group'):
        for (t1, t2) in combinations(times, 2):
            t1_df = g_df[g_df[timepoint_col] == t1][test_col]
            t2_df = g_df[g_df[timepoint_col] == t2][test_col]

            stat, p_value = stats.ranksums(
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

        stat, p_value = stats.ranksums(
            t1_df,
            t2_df, 
        )
        context = replace_underscore_dot(group) + ' ' \
            + replace_underscore_dot(timepoint_col) + ' ' + str(t1) \
            + ' vs ' + str(t2)
        print_p_value(context, p_value, show_ns=show_ns)
