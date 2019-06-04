
def y_col_type(string: str):
    if string in ['gr_percent_engraftment', 'b_percent_engraftment', 'lineage_bias']:
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