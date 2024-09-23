import os
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np

### shift

def shift_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> List[pd.Series]:
    """
    연속형 변수의 shift feature 생성
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        List[pd.Series]
    """
    df_shift_dict = [
        df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
        for conti_col in conti_cols
        for interval in intervals
    ]
    return df_shift_dict

### rolling

def rolling_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
    funcs: List[str],
    **params,
) -> pd.DataFrame:
    """
    Create rolling features
    Args:
        df (pd.DataFrame): Sorted dataframe
        conti_cols (List[str]): continuous colnames
        intervals (List[str]): rolling window widths
        funcs (List[str]): aggregation functions e.g. ["mean", "median", "max"]
        **params: more input for rolling
        Returns:
        pd.DataFrame
    """
    # 리스트 컴프리헨션 시작: 모든 조합의 rolling 특성 생성
    df_rolling_list = [
        # 지정된 연속형 변수 컬럼 선택
        df[conti_col]
        # rolling 윈도우 적용, 추가 파라미터 전달
        .rolling(interval, **params)
        # 지정된 집계 함수 적용
        .agg({f"{conti_col}": func})
        # 결과 컬럼 이름 변경
        .rename({conti_col: f"{conti_col}_{func}_{interval}"}, axis=1)
        for conti_col in conti_cols
        for interval in intervals
        for func in funcs
        ]
    # 생성된 모든 rolling 특성을 하나의 DataFrame으로 결합하여 반환
    return pd.concat(df_rolling_list, axis = 1)