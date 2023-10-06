from math import sqrt
import pandas as pd
import pytest

from mask_and_normalize import generate_std_in_meter


def test_standard_loc_std():
    df = pd.DataFrame({
        'Unnamed: 0': {3273092: 3273092},
        'level_0': {3273092: 2169821.0},
        'index': {3273092: '30791__2022-12-24'},
        'n': {3273092: 73407},
        'long_mean': {3273092: 127.967610869368},
        'lati_mean': {3273092: 35.3937421939593},
        'long_std': {3273092: 0.161735168479704},
        'lati_std': {3273092: 0.331871467014613},
        'path_sum': {3273092: 300006.921346315},
        'lati_std_in_meter': {3273092: 36943.76271688346},
        'long_std_in_meter': {3273092: 14378.853963309766},
        'loc_std': {3273092: 28032.06240164624}})

    df = generate_std_in_meter(df)
    print(df)
    result_value = sqrt(((1.617352e-01 * 88903.69) ** 2 + (3.318715e-01 * 111319.49) ** 2) / 2)
    assert pytest.approx(result_value) == df['loc_std'].tolist()[0]
