
def y_col_type(string: str):
    if string in ['myeloid_percent_abundance', 'lymphoid_percent_abundance', 'lineage_bias']:
        return string
    if string == 'None':
        return None
    raise ValueError('y_col not in available values')

def timepoint_type(string: str):
    try:
        int(string)
        return string
    except ValueError:
        if string in ['last', 'first']:
            return string
        if string == 'None':
            return None
        raise ValueError('Time point wrong type')

def change_type(string: str):
    if string.lower() in ['changed', 'unchanged', 'lymphoid', 'myeloid']:
        return string
    if string == 'None':
        return None
    raise ValueError('Invalid change type selected')