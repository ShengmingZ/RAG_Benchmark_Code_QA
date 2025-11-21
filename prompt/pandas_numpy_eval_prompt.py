import sys
sys.path.append('/home/zhaoshengming/Code_RAG_Benchmark')
from prompt.prompt_utils import ensemble_prompt

# LLAMA_SYSTEM_PROMPT = """You are a senior python programmer, given some potentially useful api documents tagged `## Potential documents`, and a unfinished code snippet tagged `## Unfinished Code Snippet`, your task is to complete the code snippet according to the comments in the code.
# You should generate the complete code snippet, and the code should start with <code> and end with </code>
# """

LLAMA_SYSTEM_PROMPT = """You are a senior python programmer. 

Input:
- useful api documents tagged `## API Documents`
- Incomplete code tagged `## Incomplete Code`.

Task:
Follow the API documents to complete the code.

Output Rules:
1. Keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""

# todo: New no ret prompt
LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer. 

Input:
- Incomplete code tagged `## Incomplete Code`.

Task:
complete the code.

Output Rules:
1. Keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""

SYSTEM_PROMPT_ZERO_SHOT = """Input:
- useful api documents tagged `## API Documents`
- Incomplete code tagged `## Incomplete Code`.

Task:
Follow the API documents to complete the code.

Output Rules:
1. Keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""

# todo: OG no ret prompt
# LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer, given a unfinished code snippet tagged `## Incomplete Code`, your task is to complete the code snippet according to the comments in the code.
# You should generate the complete code snippet without changing the existing code, and the code should in <code> and </code> tags
# """



def prompt_0shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""
## API Documents:
{potential_docs}
\n
## Incomplete Code:
{question}
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_0shot_no_ret(question, model, pads=''):
    user_prompt = f"""
{pads}\n
## Incomplete Code:
{question}
"""
    sys_prompt = LLAMA_SYSTEM_PROMPT_NO_RET
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template




def prompt_3shot(ret_docs, question, model):
    examples_3shot = '''## API Documents:
0: arange()
    arange([start,] stop[, step,], dtype=None, *, like=None)
    
    Return evenly spaced values within a given interval.
    
    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.
    
    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `numpy.linspace` for these cases.
    
    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    like : array_like
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument.
    
        .. versionadded:: 1.20.0
    
    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
    
        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
        
        
## Incomplete Code:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
```
    
    
## Complete Code, keep existing code exactly the same:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## API Documents:
0: sorted(iterable, *, key=None, reverse=False)
    Return a new list containing all items from the iterable in ascending order.
    
    A custom key function can be supplied to customize the sort order, and the
    reverse flag can be set to request the result in descending order.
    
1: isna(obj)
    Detect missing values for an array-like object.
    
    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).
    
    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.
    
    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.
    
    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.
    
    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.
    
    >>> pd.isna('dog')
    False
    
    >>> pd.isna(pd.NA)
    True
    
    >>> pd.isna(np.nan)
    True
    
    ndarrays result in an ndarray of booleans.
    
    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])
    
    For indexes, an ndarray of booleans is returned.
    
    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])
    
    For Series and DataFrame, the same type is returned, containing booleans.
    
    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1 

2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
    Apply a function along an axis of the DataFrame.
    
    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.
    
    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:
    
        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.
    
    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:
    
        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.
    
    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):
    
        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.
    
        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.
    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to
        `func`.
    
    Returns
    -------
    Series or DataFrame
    
3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)
    Remove missing values.
    
    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.
    
    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.
    
        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.
    
        .. versionchanged:: 1.0.0
    
           Pass tuple or list to drop on multiple axes.
           Only a single axis is allowed.
    
    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.
    
        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.
    
    thresh : int, optional
        Require that many non-NA values.
    subset : array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.
    inplace : bool, default False
        If True, do operation inplace and return None.
    
    Returns
    -------
    DataFrame or None
        DataFrame with NA entries dropped from it or None if ``inplace=True``.
    
    See Also
    --------
    DataFrame.isna: Indicate missing values.
    DataFrame.notna : Indicate existing (non-missing) values.
    DataFrame.fillna : Replace missing values.
    Series.dropna : Drop missing values.
    Index.dropna : Drop missing indices.
    
    Examples
    --------
    >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),
    ...                             pd.NaT]})
    >>> df
           name        toy       born
           
           
## Incomplete Code:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =
```


## Complete Code, keep existing code exactly the same:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df = df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## API Documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)
    Render a DataFrame to a console-friendly tabular output.
    
    Parameters
    ----------
    buf : str, Path or StringIO-like, optional, default None
        Buffer to write to. If None, the output is returned as a string.
    columns : sequence, optional, default None
        The subset of columns to write. Writes all columns by default.
    col_space : int, list or dict of int, optional
        The minimum width of each column.
    header : bool or sequence, optional
        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.
    index : bool, optional, default True
        Whether to print index (row) labels.
    na_rep : str, optional, default 'NaN'
        String representation of ``NaN`` to use.
    formatters : list, tuple or dict of one-param. functions, optional
        Formatter functions to apply to columns' elements by position or
        name.
        The result of each function must be a unicode string.
        List/tuple must be of length equal to the number of columns.
    float_format : one-parameter function, optional, default None
        Formatter function to apply to columns' elements if they are
        floats. This function must return a unicode string and will be
        applied only to the non-``NaN`` elements, with ``NaN`` being
        
        
## Incomplete Code:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =
```


## Complete Code, keep existing code exactly the same:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string = df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""{examples_3shot}
\n\n
## API Documents:
{potential_docs}
\n
## Incomplete Code:
```
{question}
```
\n
## Complete Code, keep existing code exactly the same:
"""
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt



if __name__ == '__main__':
    example_doc = "to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)\n    Render a DataFrame to a console-friendly tabular output.\n    \n    Parameters\n    ----------\n    buf : str, Path or StringIO-like, optional, default None\n        Buffer to write to. If None, the output is returned as a string.\n    columns : sequence, optional, default None\n        The subset of columns to write. Writes all columns by default.\n    col_space : int, list or dict of int, optional\n        The minimum width of each column.\n    header : bool or sequence, optional\n        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.\n    index : bool, optional, default True\n        Whether to print index (row) labels.\n    na_rep : str, optional, default 'NaN'\n        String representation of ``NaN`` to use.\n    formatters : list, tuple or dict of one-param. functions, optional\n        Formatter functions to apply to columns' elements by position or\n        name.\n        The result of each function must be a unicode string.\n        List/tuple must be of length equal to the number of columns.\n    float_format : one-parameter function, optional, default None\n        Formatter function to apply to columns' elements if they are\n        floats. This function must return a unicode string and will be\n        applied only to the non-``NaN`` elements, with ``NaN`` being\n        handled by ``na_rep``.\n    \n        .. versionchanged:: 1.2.0\n    \n    sparsify : bool, optional, default True\n        Set to False for a DataFrame with a hierarchical index to print\n        every multiindex key at each row.\n    index_names : bool, optional, default True\n        Prints the names of the indexes.\n    justify : str, default None\n        How to justify the column labels. If None uses the option from\n        the print configuration (controlled by set_option), 'right' out\n        of the box. Valid values are\n    \n        * left\n        * right\n        * center\n        * justify\n        * justify-all\n        * start\n        * end\n        * inherit\n        * match-parent\n        * initial\n        * unset.\n    max_rows : int, optional\n        Maximum number of rows to display in the console.\n    min_rows : int, optional\n        The number of rows to display in the console in a truncated repr\n        (when number of rows is above `max_rows`).\n    max_cols : int, optional\n        Maximum number of columns to display in the console.\n    show_dimensions : bool, default False\n        Display DataFrame dimensions (number of rows by number of columns).\n    decimal : str, default '.'\n        Character recognized as decimal separator, e.g. ',' in Europe.\n    \n    line_width : int, optional\n        Width to wrap a line in characters.\n    max_colwidth : int, optional\n        Max width to truncate each column in characters. By default, no limit.\n    \n        .. versionadded:: 1.0.0\n    encoding : str, default \"utf-8\"\n        Set character encoding.\n    \n        .. versionadded:: 1.0\n    \n    Returns\n    -------\n    str or None\n        If buf is None, returns the result as a string. Otherwise returns\n        None.\n    \n    See Also\n    --------\n    to_html : Convert DataFrame to HTML.\n    \n    Examples\n    --------\n    >>> d = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}\n    >>> df = pd.DataFrame(d)\n    >>> print(df.to_string())\n       col1  col2\n    0     1     4\n    1     2     5\n    2     3     6\n\n"

    from generator.generate_utils import truncate_docs
    print(truncate_docs([example_doc], 'gpt-3.5-turbo-0125', 500)[0])







def prompt_cot(ret_docs, question, model):
    examples_cot = '''## API Documents:
0: arange()
    arange([start,] stop[, step,], dtype=None, *, like=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `numpy.linspace` for these cases.

    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    like : array_like
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument.

        .. versionadded:: 1.20.0

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.


## Incomplete Code:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
```


According to the problem, I need to create sliding windows of length L with stride S from array a.
First, I need to create starting positions for each window - these are at positions 0, S, 2*S, etc., which I can generate using S*np.arange(nrows).
Then, for each starting position, I need L consecutive indices as offsets, which I can create using np.arange(L).
Finally, I can use broadcasting by reshaping the starting positions into a column vector with [:,None] and adding the offset array to get all required indices.
Based on this, I implement the complete code, and keep existing code unchanged:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## API Documents:
0: sorted(iterable, *, key=None, reverse=False)
    Return a new list containing all items from the iterable in ascending order.

    A custom key function can be supplied to customize the sort order, and the
    reverse flag can be set to request the result in descending order.

1: isna(obj)
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna('dog')
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1 

2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.

    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.

    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:

        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.
    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    Series or DataFrame

3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)
    Remove missing values.

    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.

        .. versionchanged:: 1.0.0

           Pass tuple or list to drop on multiple axes.
           Only a single axis is allowed.

    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    thresh : int, optional
        Require that many non-NA values.
    subset : array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.
    inplace : bool, default False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with NA entries dropped from it or None if ``inplace=True``.

    See Also
    --------
    DataFrame.isna: Indicate missing values.
    DataFrame.notna : Indicate existing (non-missing) values.
    DataFrame.fillna : Replace missing values.
    Series.dropna : Drop missing values.
    Index.dropna : Drop missing indices.

    Examples
    --------
    >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),
    ...                             pd.NaT]})
    >>> df
           name        toy       born


## Incomplete Code:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =
```


According to the instruction, I need to use sorted to align non-NULL data at the top, then use dropna to drop all rows with all NaN.
The document shows apply() for column-wise operations, sorted() with key parameter for custom sorting, and dropna() for removing rows with all NaN values.
First, I should apply sorted() to each column with pd.isnull as key to move non-null values to the top, since False (non-null) sorts before True (null).
Then, I should use dropna() with how='all' to remove rows where all values are NaN.
Based on this, I implement the complete code, and keep existing code unchanged:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df = df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## API Documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)
    Render a DataFrame to a console-friendly tabular output.

    Parameters
    ----------
    buf : str, Path or StringIO-like, optional, default None
        Buffer to write to. If None, the output is returned as a string.
    columns : sequence, optional, default None
        The subset of columns to write. Writes all columns by default.
    col_space : int, list or dict of int, optional
        The minimum width of each column.
    header : bool or sequence, optional
        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.
    index : bool, optional, default True
        Whether to print index (row) labels.
    na_rep : str, optional, default 'NaN'
        String representation of ``NaN`` to use.
    formatters : list, tuple or dict of one-param. functions, optional
        Formatter functions to apply to columns' elements by position or
        name.
        The result of each function must be a unicode string.
        List/tuple must be of length equal to the number of columns.
    float_format : one-parameter function, optional, default None
        Formatter function to apply to columns' elements if they are
        floats. This function must return a unicode string and will be
        applied only to the non-``NaN`` elements, with ``NaN`` being


## Incomplete Code:
```
import pandas as pd
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =
```



According to the problem, I need to convert the DataFrame to a string representation without showing the index.
The document shows to_string() which renders a DataFrame to console-friendly tabular output.
I need to use the index parameter to control whether the row index is displayed - setting index=False will hide it.
Based on this, I implement the complete code, and keep existing code unchanged:
```
import pandas as pd
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string = df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""{examples_cot}
\n\n
## API Documents:
{potential_docs}
\n
## Incomplete Code:
```
{question}
```
\n
"""
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt




def prompt_least_to_most(ret_docs, question, model):
    examples_3shot = '''## API Documents:
0: arange()
    arange([start,] stop[, step,], dtype=None, *, like=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `numpy.linspace` for these cases.

    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    like : array_like
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument.

        .. versionadded:: 1.20.0

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.


## Incomplete Code:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
```


Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: How to generate starting indices for each sliding window?
I can use S*np.arange(nrows) to create starting positions [0, S, 2*S, ...].

Sub problem 2: How to generate offset indices within each window?
I can use np.arange(L) to create offsets [0, 1, 2, ..., L-1] for each window.

Sub problem 3: How to combine starting positions with offsets using broadcasting?
I can reshape starting positions to column vector with [:,None] and add the offset array.

Now combining all solutions to generate the complete code, and keep existing code unchanged:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## API Documents:
0: sorted(iterable, *, key=None, reverse=False)
    Return a new list containing all items from the iterable in ascending order.

    A custom key function can be supplied to customize the sort order, and the
    reverse flag can be set to request the result in descending order.

1: isna(obj)
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna('dog')
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1 

2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.

    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.

    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:

        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.
    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    Series or DataFrame

3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)
    Remove missing values.

    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.

        .. versionchanged:: 1.0.0

           Pass tuple or list to drop on multiple axes.
           Only a single axis is allowed.

    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    thresh : int, optional
        Require that many non-NA values.
    subset : array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.
    inplace : bool, default False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with NA entries dropped from it or None if ``inplace=True``.

    See Also
    --------
    DataFrame.isna: Indicate missing values.
    DataFrame.notna : Indicate existing (non-missing) values.
    DataFrame.fillna : Replace missing values.
    Series.dropna : Drop missing values.
    Index.dropna : Drop missing indices.

    Examples
    --------
    >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),
    ...                             pd.NaT]})
    >>> df
           name        toy       born


## Incomplete Code:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =
```


Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: How to move non-null values to the top of each column?
I can use apply() with sorted() and pd.isnull as key, since False (non-null) sorts before True (null).

Sub problem 2: How to remove rows that are completely empty?
I can use dropna() with how='all' parameter to remove rows where all values are NaN.

Now combining all solutions to generate the complete code, and keep existing code unchanged:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df = df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how='all')
```




## API Documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)
    Render a DataFrame to a console-friendly tabular output.

    Parameters
    ----------
    buf : str, Path or StringIO-like, optional, default None
        Buffer to write to. If None, the output is returned as a string.
    columns : sequence, optional, default None
        The subset of columns to write. Writes all columns by default.
    col_space : int, list or dict of int, optional
        The minimum width of each column.
    header : bool or sequence, optional
        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.
    index : bool, optional, default True
        Whether to print index (row) labels.
    na_rep : str, optional, default 'NaN'
        String representation of ``NaN`` to use.
    formatters : list, tuple or dict of one-param. functions, optional
        Formatter functions to apply to columns' elements by position or
        name.
        The result of each function must be a unicode string.
        List/tuple must be of length equal to the number of columns.
    float_format : one-parameter function, optional, default None
        Formatter function to apply to columns' elements if they are
        floats. This function must return a unicode string and will be
        applied only to the non-``NaN`` elements, with ``NaN`` being


## Incomplete Code:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =
```


Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: How to convert DataFrame to string format?
I can use to_string() method which renders DataFrame to console-friendly tabular output.

Sub problem 2: How to hide the index when converting to string?
I can use the index parameter in to_string() and set it to False to exclude the index.

Now combining all solutions to generate the complete code, and keep existing code unchanged:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string = df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""{examples_3shot}
\n\n
## API Documents:
{potential_docs}
\n
## Incomplete Code:
```
{question}
```
\n
"""
    if 'gpt' in model:
        prompt = [dict(role='user', content=user_prompt)]
    else:
        prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt


def prompt_self_refine(ret_docs, question, model, initial_output):
    examples_3shot = '''## API Documents:
0: arange()
    arange([start,] stop[, step,], dtype=None, *, like=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `numpy.linspace` for these cases.

    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    like : array_like
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument.

        .. versionadded:: 1.20.0

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.


## Incomplete Code:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
```

## Initial Code Solution:
```
import numpy as np
def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    result = []
    for i in range(nrows):
        result.append(a[i*S:i*S+L])
    return np.array(result)
```

Please first provide feedback on this solution and then refine it based on the feedback.



Feedback: The current solution uses a loop to extract each window, which is inefficient and not leveraging numpy's vectorization capabilities.
We can use advanced indexing with broadcasting to create all windows at once by combining starting positions with window offsets.

Refined solution:
```
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## API Documents:
0: sorted(iterable, *, key=None, reverse=False)
    Return a new list containing all items from the iterable in ascending order.

    A custom key function can be supplied to customize the sort order, and the
    reverse flag can be set to request the result in descending order.

1: isna(obj)
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays, ``NaT`` in datetimelike).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    notna : Boolean inverse of pandas.isna.
    Series.isna : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> pd.isna('dog')
    False

    >>> pd.isna(pd.NA)
    True

    >>> pd.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> pd.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For indexes, an ndarray of booleans is returned.

    >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,
    ...                           "2017-07-08"])
    >>> index
    DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],
                  dtype='datetime64[ns]', freq=None)
    >>> pd.isna(index)
    array([False, False,  True, False])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
    >>> df
         0     1 

2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
    Apply a function along an axis of the DataFrame.

    Objects passed to the function are Series objects whose index is
    either the DataFrame's index (``axis=0``) or the DataFrame's columns
    (``axis=1``). By default (``result_type=None``), the final return type
    is inferred from the return type of the applied function. Otherwise,
    it depends on the `result_type` argument.

    Parameters
    ----------
    func : function
        Function to apply to each column or row.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis along which the function is applied:

        * 0 or 'index': apply function to each column.
        * 1 or 'columns': apply function to each row.

    raw : bool, default False
        Determines if row or column is passed as a Series or ndarray object:

        * ``False`` : passes each row or column as a Series to the
          function.
        * ``True`` : the passed function will receive ndarray objects
          instead.
          If you are just applying a NumPy reduction function this will
          achieve much better performance.

    result_type : {'expand', 'reduce', 'broadcast', None}, default None
        These only act when ``axis=1`` (columns):

        * 'expand' : list-like results will be turned into columns.
        * 'reduce' : returns a Series if possible rather than expanding
          list-like results. This is the opposite of 'expand'.
        * 'broadcast' : results will be broadcast to the original shape
          of the DataFrame, the original index and columns will be
          retained.

        The default behaviour (None) depends on the return value of the
        applied function: list-like results will be returned as a Series
        of those. However if the apply function returns a Series these
        are expanded to columns.
    args : tuple
        Positional arguments to pass to `func` in addition to the
        array/series.
    **kwargs
        Additional keyword arguments to pass as keywords arguments to
        `func`.

    Returns
    -------
    Series or DataFrame

3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)
    Remove missing values.

    See the :ref:`User Guide <missing_data>` for more on which values are
    considered missing, and how to work with missing data.

    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed.

        * 0, or 'index' : Drop rows which contain missing values.
        * 1, or 'columns' : Drop columns which contain missing value.

        .. versionchanged:: 1.0.0

           Pass tuple or list to drop on multiple axes.
           Only a single axis is allowed.

    how : {'any', 'all'}, default 'any'
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.

        * 'any' : If any NA values are present, drop that row or column.
        * 'all' : If all values are NA, drop that row or column.

    thresh : int, optional
        Require that many non-NA values.
    subset : array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.
    inplace : bool, default False
        If True, do operation inplace and return None.

    Returns
    -------
    DataFrame or None
        DataFrame with NA entries dropped from it or None if ``inplace=True``.

    See Also
    --------
    DataFrame.isna: Indicate missing values.
    DataFrame.notna : Indicate existing (non-missing) values.
    DataFrame.fillna : Replace missing values.
    Series.dropna : Drop missing values.
    Index.dropna : Drop missing indices.

    Examples
    --------
    >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
    ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],
    ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),
    ...                             pd.NaT]})
    >>> df
           name        toy       born


## Incomplete Code:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =
```


## Initial Code Solution:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df = df.dropna().apply(lambda x: sorted(x))
```

Please first provide feedback on this solution and then refine it based on the feedback.



Feedback: The current solution has two issues: 
1) dropna() removes entire rows with any NaN values, which loses valid data from other columns
2) sorted(x) without a key parameter sorts by value, not by null status. 
We need to sort within each column to move non-null values to the top, then remove only completely empty rows.

Refined solution:
```
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df = df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## API Documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)
    Render a DataFrame to a console-friendly tabular output.

    Parameters
    ----------
    buf : str, Path or StringIO-like, optional, default None
        Buffer to write to. If None, the output is returned as a string.
    columns : sequence, optional, default None
        The subset of columns to write. Writes all columns by default.
    col_space : int, list or dict of int, optional
        The minimum width of each column.
    header : bool or sequence, optional
        Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.
    index : bool, optional, default True
        Whether to print index (row) labels.
    na_rep : str, optional, default 'NaN'
        String representation of ``NaN`` to use.
    formatters : list, tuple or dict of one-param. functions, optional
        Formatter functions to apply to columns' elements by position or
        name.
        The result of each function must be a unicode string.
        List/tuple must be of length equal to the number of columns.
    float_format : one-parameter function, optional, default None
        Formatter function to apply to columns' elements if they are
        floats. This function must return a unicode string and will be
        applied only to the non-``NaN`` elements, with ``NaN`` being


## Incomplete Code:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =
```


## Initial Code Solution:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string = str(df)
```

Please first provide feedback on this solution and then refine it based on the feedback.



Feedback: The current solution uses str(df) which converts the DataFrame to string but still includes the index column.
This doesn't meet the requirement of excluding the index. We need to use to_string() method with index=False parameter to specifically control the index display.

Refined solution:
```
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string = df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""{examples_3shot}
\n\n
## API Documents:
{potential_docs}
\n
## Incomplete Code:
```
{question}
```
\n
## Initial Code Solution:
```
{initial_output}
```

Please first provide feedback on this solution and then refine it based on the feedback.
"""
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt



def prompt_emotion(ret_docs, question, model):
    system_prompt_emotion = "This is very important to my career.\n\n" + SYSTEM_PROMPT_ZERO_SHOT
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Incomplete Code: 
{question}
"""
    prompt_template = ensemble_prompt(sys_prompt=system_prompt_emotion, user_prompt=user_prompt, model=model)
    return prompt_template



def prompt_zero_shot_cot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""
## API Documents:
{potential_docs}
\n
## Incomplete Code:
{question}

Let's think it step by step.
"""

    prompt_template = ensemble_prompt(sys_prompt=SYSTEM_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template




def prompt_plan_and_solve(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""
## API Documents:
{potential_docs}
\n
## Incomplete Code:
{question}

Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
"""

    prompt_template = ensemble_prompt(sys_prompt=SYSTEM_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template




def prompt_con(ret_docs, question, model):
    sys_prompt = """Input:
- useful api documents tagged `## API Documents`
- Incomplete code tagged `## Incomplete Code`.

Task:
Follow the API documents to complete the code, by:
1. Reading the program description and API documents to gather relevant information
2. Writing brief reading notes summarizing key points from the API documents
3. Assessing the relevance between the program requirements and available API functions
4. Using relevant API functions to complete the python function, or implementing without the given APIs if none are relevant

Output Format:
Reading notes: [Brief summary of key API functions and their purposes]
Relevance assessment: [How the APIs relate to the program requirements]
```
[Complete python function with existing code unchanged]
```
"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""
## API Documents:
{potential_docs}
\n
## Incomplete Code:
{question}
"""

    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template





# def prompt_RaR(ret_docs, question, model):
#     RaR_prompt = """You are a senior python programmer, given some Potential Documents and a Uncompleted Code Snippet
# Your task is to first rephrase and expand the Problem, then complete the code snippet"""
#
#     potential_docs = ''
#     for idx, ret_doc in enumerate(ret_docs):
#         potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
#     user_prompt = f"""
# ## Potential documents:
# {potential_docs}
# \n
# ## Uncompleted Code Snippet:
# {question}
# """
#     sys_prompt = RaR_prompt
#     prompt = ensemble_prompt(sys_prompt, user_prompt, model)
#     # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
#     return prompt



# if __name__ == '__main__':
#     """get random samples as few shot examples"""
#     import sys, platform
#     import random
#     import json
#
#     system = platform.system()
#     if system == 'Darwin':
#         root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
#     elif system == 'Linux':
#         root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
#     sys.path.insert(0, root_path)
#     from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
#     from dataset_utils.corpus_utils import PythonDocsLoader
#     from generator.generate_utils import truncate_docs
#
#
#     def get_few_shots(k):
#         loader = PandasNumpyEvalLoader()
#         qs_list = loader.load_qs_list()
#         qs_id_list = [qs['qs_id'] for qs in qs_list]
#         dataset = json.load(open(loader.data_file))
#         unincluded_dataset = []
#         for data in dataset:
#             if data['task_id'] not in qs_id_list:
#                 unincluded_dataset.append(data)
#         random.seed()
#         few_shots = random.sample(unincluded_dataset, k=k)
#         for shot in few_shots:
#             print(shot['prompt'])
#             print(shot['canonical_solution'])
#             print(shot.keys())
#
#
#     # get_few_shots(k=3)
#
#
#     '''
# import numpy as np
#
# def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
#     """
#     I want to create a matrix of sub sequences from this array of length L with stride S.
#     Return the numpy array of sub sequences.
#     """
#     nrows = ((a.size-L)//S)+1
#
# ['    return a[S*np.arange(nrows)[:,None] + np.arange(L)]']
#
#     '''
#
#     '''
# import pandas as pd
# import numpy as np
# df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# # Move next value to first empty row pandas
# # how do i move each value from a column to the first empty "row/cell" in pandas?
# # use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
# new_df =
# [" df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
#     '''
#
#     '''
#     df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# # How to obtain pandas DataFrame without index
# # I want to print the whole dataframe, but I don't want to print the index
# df_string =
# [' df.to_string(index=False)']
#     '''
#
#     api_signs = [['matplotlib.pylab.arange'], ["builtins.sorted", "pandas.core.common.isna", "pandas.core.frame.DataFrame.apply", "pandas.core.frame.DataFrame.dropna"], ["pandas.core.frame.DataFrame.to_string"]]
#     for api_sign in api_signs:
#         docs = PythonDocsLoader().get_docs(api_sign)
#         docs = truncate_docs(docs, model='codellama-13b-instruct', max_length=1000)
#         potential_docs = ''
#         for idx, ret_doc in enumerate(docs):
#             potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
#         print(potential_docs)
