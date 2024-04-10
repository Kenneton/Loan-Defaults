from pydantic import  create_model
from typing import Optional
import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype, is_bool_dtype, is_object_dtype

def pandas_type_to_python_type(pandas_type):

    if is_float_dtype(pandas_type):
        return (Optional[float], None)
    elif is_integer_dtype(pandas_type):
        return (Optional[int], None)
    elif is_bool_dtype(pandas_type):
        return (Optional[bool], None)
    elif is_object_dtype(pandas_type) or isinstance(pandas_type, pd.CategoricalDtype):
        return (Optional[str], None)
    else:
        return (Optional[str], None)


dtypes = pd.read_pickle("dtypes.pkl") 

fields = {col: pandas_type_to_python_type(dtype) for col, dtype in dtypes.items()}
DynamicModel = create_model('DynamicModel', **fields)
