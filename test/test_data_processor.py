import pandas as pd
import pytest
from app.data_processor import align_dataframes_by_time, calculate_w_a_difference


def test_no_gases_list_raises_value_error():
    df = pd.DataFrame({
        'RAW_ADC_NO_W': [1],
        'RAW_ADC_NO_A': [0]
    })
    with pytest.raises(ValueError) as e:
        calculate_w_a_difference(df, [])
    assert str(e.value) == "No gases in list"


def test_empty_dataframe_raises_value_error():
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError) as e:
        calculate_w_a_difference(df_empty, ['NO'])
    assert "dataframe is empty" in str(e.value)


def test_basic_NO_CO_difference():
    df = pd.DataFrame({
        'RAW_ADC_NO_W': [10, 20, 30],
        'RAW_ADC_NO_A': [1,  2,  3],
        'RAW_ADC_CO_W': [100, 200, 300],
        'RAW_ADC_CO_A': [10,  20,  30],
        'other': [0, 0, 0]
    })
    # Work on a copy so we can test identity
    df_copy = df.copy()
    result = calculate_w_a_difference(df_copy, ['NO', 'CO'])

    # should modify in-place and return the same object
    assert result is df_copy
    expected_cols = {'other', 'NO_W_A', 'CO_W_A'}
    assert set(result.columns) == expected_cols

    assert result['NO_W_A'].tolist() == [9, 18, 27]
    assert result['CO_W_A'].tolist() == [90, 180, 270]


def test_align_dataframes_by_time():
    # df1: 7 days from Jan 1–7, df2: 7 days from Jan 4–10 → overlap Jan 4–7
    df1 = pd.DataFrame(data={'v1': list(range(7))}, index=pd.date_range('2021-01-01', periods=7, freq='D'))
    df2 = pd.DataFrame(data={'v2': list(range(7, 14))}, index=pd.date_range('2021-01-04', periods=7, freq='D'))

    a1, a2 = align_dataframes_by_time(df1.copy(), df2.copy())

    expected_idx = pd.date_range('2021-01-04', '2021-01-07', freq='D')
    pd.testing.assert_index_equal(a1.index, expected_idx)
    pd.testing.assert_index_equal(a2.index, expected_idx)
    assert a1['v1'].tolist() == [3, 4, 5, 6]
    assert a2['v2'].tolist() == [7, 8, 9, 10]


def test_align_dataframes_by_time_returns_empty_dataframes():
    df1 = pd.DataFrame(data={'v1': [1, 2, 3]}, index=pd.date_range('2021-01-01', periods=3, freq='D'))
    df2 = pd.DataFrame(data= {'v2': [4, 5, 6]}, index=pd.date_range('2021-01-10', periods=3, freq='D'))

    a1, a2 = align_dataframes_by_time(df1, df2)

    assert isinstance(a1, pd.DataFrame)
    assert isinstance(a2, pd.DataFrame)
    assert a1.empty
    assert a2.empty


def test_align_dataframes_by_time_string_and_datetime_index_conversion():
    # df1 index as strings, df2 as Timestamps
    df1 = pd.DataFrame(data={'v1': [10, 20, 30]}, index=['2021-06-01', '2021-06-02', '2021-06-03'])
    df2 = pd.DataFrame(data={'v2': [100, 200, 300]}, index=pd.to_datetime(['2021-06-02', '2021-06-03', '2021-06-04']))

    a1, a2 = align_dataframes_by_time(df1, df2)

    expected = pd.DatetimeIndex(['2021-06-02', '2021-06-03'])
    pd.testing.assert_index_equal(a1.index, expected)
    pd.testing.assert_index_equal(a2.index, expected)

    assert isinstance(a1.index, pd.DatetimeIndex)
    assert isinstance(a2.index, pd.DatetimeIndex)

    assert a1['v1'].tolist() == [20, 30]
    assert a2['v2'].tolist() == [100, 200]


def test_align_dataframes_by_time_empty_input_dataframes():
    # both have columns but no rows
    df1 = pd.DataFrame(columns=['a'], index=[])
    df2 = pd.DataFrame(columns=['b'], index=[])

    a1, a2 = align_dataframes_by_time(df1, df2)

    assert a1.empty and a2.empty
    assert list(a1.columns) == ['a']
    assert list(a2.columns) == ['b']


def test_align_dataframes_by_time_invalid_datetime_strings_raises():
    # df1 has a non–parseable date
    df1 = pd.DataFrame({'x': [1]}, index=['not-a-date'])
    df2 = pd.DataFrame({'y': [2]}, index=['2021-01-01'])

    with pytest.raises(ValueError):
        align_dataframes_by_time(df1, df2)

