""" Functions to calculate lineage bias given FACS / cell count data

"""
import pandas as pd
from math import pi, sin, atan
import numpy as np

def calc_bias(m_value: float, l_value: float) -> float:
    if l_value == 0.0:
        if m_value == 0.0:
            towards_myloid_angle = pi/4
        else:
            towards_myloid_angle = pi/2
    else:
        towards_myloid_angle = atan(m_value/l_value)

    lineage_bias = sin(2 * (towards_myloid_angle - (pi/4)))
    return lineage_bias



def calc_aging_lineage_bias(
    gr_abundance: float,
    b_abundance: float,
    mouse_id: str,
) -> float:
    """ Calculates lineage bias accounting and
    does filtering/normalization

    Args:
        gr_abundance (float)
        b_abundance (float)
        mouse_id (str)

    Returns:
        float: lineage bias value from -1 to 1
    """
    filters = pd.read_csv('FACS_filters_aging.csv')
    filt = filters[filters.mouse_id == mouse_id]
    filt_bool = (
        (gr_abundance < filt['gr_filter'].values[0]) &
        (b_abundance < filt['b_filter'].values[0])
    )
    if filt_bool:
        return np.nan
    norm_gr = gr_abundance/filt['gr'].values[0]
    norm_b = b_abundance/filt['b'].values[0]
    return calc_bias(norm_gr, norm_b)

