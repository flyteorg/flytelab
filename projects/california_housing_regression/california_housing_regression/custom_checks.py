from typing import Optional

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pandera as pa
import pandera.extensions as extensions
import scipy.stats as stats
from pandera.strategies import pandas_dtype_strategy


def mean_eq(pandas_obj, *, value, alpha):
    """
    Null hypothesis: the mean of data is equal to the value argument.
    If pvalue is greater than alpha, we can't reject the null hypothesis
    """
    _, pvalue = stats.ttest_1samp(pandas_obj, value)
    return pvalue >= alpha


def mean_eq_strategy(
    pandera_dtype: pa.DataType,
    strategy: Optional[st.SearchStrategy] = None,
    *,
    value,
    alpha,
):
    if strategy:
        raise pa.errors.BaseStrategyOnlyError(
            "mean_eq_strategy is a base strategy. You cannot specify the "
            "strategy argument to chain it to a parent strategy."
        )
    return pandas_dtype_strategy(
        pandera_dtype,
        strategy=st.builds(lambda: np.random.normal(loc=value, scale=0.01))
    )


extensions.register_check_method(
    mean_eq,
    statistics=["value", "alpha"],
    strategy=mean_eq_strategy,
    supported_types=[pd.Series],
    check_type="vectorized",
)
