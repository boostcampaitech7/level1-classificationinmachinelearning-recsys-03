import os
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np

class DataPreprocessing:
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