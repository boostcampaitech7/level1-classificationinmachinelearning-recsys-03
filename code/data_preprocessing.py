import os
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from scipy import stats

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
        df[conti_col] # 지정된 연속형 변수 컬럼 선택
        .rolling(interval, **params) # rolling 윈도우 적용, 추가 파라미터 전달
        .agg({f"{conti_col}": func}) # 지정된 집계 함수 적용
        .rename({conti_col: f"{conti_col}_{func}_{interval}"}, axis=1) # 결과 컬럼 이름 변경
        for conti_col in conti_cols
        for interval in intervals
        for func in funcs
    ]
    return pd.concat(df_rolling_list, axis = 1) # 생성된 모든 rolling 특성을 하나의 DataFrame으로 결합하여 반환

### fill

def fill_feature(
    df: pd.DataFrame,
    method: str,
) -> List[pd.Series]:
    """
    missingvalue_보간
    Args:
        df (pd.DataFrame):
        method (str): mean or median
    Returns:
        List[pd.Series]:
    """
    # 숫자형 변수만 선별
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if method=="mean":      # 평균
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    if method=="median":    # 중앙값
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

### normalization

def standardize_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    데이터를 평균이 0이고 표준편차가 1이 되도록 변환 (Standard Scaling, Z-score Normalization)
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    scaler = StandardScaler() # Standard Scaler 객체 생성
    scaled_data = scaler.fit_transform(df[conti_cols]) # 데이터 스케일링 적용
    scaled_df = pd.DataFrame(scaled_data, columns=conti_cols)

    return scaled_df

def normalize_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    데이터를 [0, 1] 범위로 변환 (Min-Max Scaling)
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    scaler = MinMaxScaler() # Min-Max Scaler 객체 생성
    scaled_data = scaler.fit_transform(df[conti_cols]) # 데이터 스케일링 적용
    scaled_df = pd.DataFrame(scaled_data, columns=conti_cols)

    return scaled_df

def log_transform_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    로그 함수를 적용해 *양수*인 데이터 분포 정규화 (0과 음수는 NaN 처리)
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    trans_df = df[conti_cols].copy()

    # 로그 변환 적용: 음수 및 0은 NaN으로 처리
    for col in conti_cols:
        trans_df[col] = np.where(trans_df[col] > 0, np.log(trans_df[col]), np.nan)

    return trans_df

def box_cox_transform_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    Box-Cox 변환 적용해 *양수*인 데이터 분포 정규화 (전처리 후 사용 필요)
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    trans_df = df[conti_cols].copy()

    # Box-Cox 변환 적용
    for col in conti_cols:
        if (trans_df[col] > 0).all():
            trans_df[col] = stats.boxcox(trans_df[col])[0]

    return trans_df

def power_transform_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    거듭제곱 변환
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    transformer = PowerTransformer() # Power Transformer 객체 생성
    trans_data = transformer.fit_transform(df[conti_cols]) # 거듭제곱 변환
    trans_df = pd.DataFrame(trans_data, columns=conti_cols, index=df.index)

    return trans_df

def quantile_transform_feature(
    df: pd.DataFrame,
    conti_cols: List[str]
) -> pd.DataFrame:
    """
    분위수 변환
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
    Returns:
        pd.DataFrame
    """
    # Quantile Transformer 객체 생성
    transformer = QuantileTransformer(output_distribution='normal') # output_distribution은 어떤 분포를 따를 지
    trans_data = transformer.fit_transform(df[conti_cols])
    trans_df = pd.DataFrame(trans_data, columns=conti_cols, index=df.index)

    return trans_df