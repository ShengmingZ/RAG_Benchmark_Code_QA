from prompt.prompt_utils import ensemble_prompt

LLAMA_SYSTEM_PROMPT = """You are a senior python programmer, given some potential api documents starts with `## Potential documents`, and a unfinished code snippet starts with `## Unfinished Code Snippet`, 
you should first read the potential documents, and then use the knowledge in documents to complete the code snippet according to the comments in the code.
you should only output the uncompleted part of the code snippet, and the output code should starts with <code> and ends with </code>
"""

LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer, given a unfinished code snippet starts with `## Unfinished Code Snippet`, you need to complete the code snippet according to the comments in the code.
you should only output the uncompleted part of the code snippet, and the output code should starts with <code> and ends with </code>
"""

SYS_PROMPT_LEAST_TO_MOST = """Follow the examples to solve the last problem"""

def prompt_cot(ret_docs, question, model, existing_output=None):
    examples_prompt = '''
## Potential documents:
0: arange()     arange([start,] stop[, step,], dtype=None, *, like=None)          Return evenly spaced values within a given interval.          Values are generated within the half-open interval ``[start, stop)``     (in other words, the interval including `start` but excluding `stop`).     For integer arguments the function is equivalent to the Python built-in     `range` function, but returns an ndarray rather than a list.          When using a non-integer step, such as 0.1, the results will often not     be consistent.  It is better to use `numpy.linspace` for these cases.          Parameters     ----------     start : integer or real, optional         Start of interval.  The interval includes this value.  The default         start value is 0.     stop : integer or real         End of interval.  The interval does not include this value, except         in some cases where `step` is not an integer and floating point         round-off affects the length of `out`.     step : integer or real, optional         Spacing between values.  For any output `out`, this is the distance         between two adjacent values, ``out[i+1] - out[i]``.  The default         step size is 1.  If `step` is specified as a position argument,         `start` must also be given.     dtype : dtype         The type of the output array.  If `dtype` is not given, infer the data         type from the other input arguments.     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     arange : ndarray         Array of evenly spaced values.              For floating point arguments, the length of the result is         ``ceil((stop - start)/step)``.  Because of floating point overflow,         this rule may result in the last element of `out` being greater         than `stop`.          See Also     --------     numpy.linspace : Evenly spaced numbers with careful handling of endpoints.     numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.     numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.          Examples     --------     >>> np.arange(3)     array([0, 1, 2])     >>> np.arange(3.0)     array([ 0.,  1.,  2.])     >>> np.arange(3,7)     array([3, 4, 5, 6])     >>> np.arange(3,7,2)     array([3, 5])  

## Unfinished Code Snippet: 
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    
## Completion:
To create the matrix, we need to construct the required indices.
To construct the indices, we can first use function `np.arange` to construct the indices for first row: `np.arange(L)` and the indices for first column: `S*np.arange(nrows)[:,None]`,
then we can use broadcasting operation to construct the required indices
So the completion code is
```
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## Potential documents:
0: sorted(iterable, *, key=None, reverse=False)     Return a new list containing all items from the iterable in ascending order.          A custom key function can be supplied to customize the sort order, and the     reverse flag can be set to request the result in descending order.  
1: isna(obj)     Detect missing values for an array-like object.          This function takes a scalar or array-like object and indicates     whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``     in object arrays, ``NaT`` in datetimelike).          Parameters     ----------     obj : scalar or array-like         Object to check for null or missing values.          Returns     -------     bool or array-like of bool         For scalar input, returns a scalar boolean.         For array input, returns an array of boolean indicating whether each         corresponding element is missing.          See Also     --------     notna : Boolean inverse of pandas.isna.     Series.isna : Detect missing values in a Series.     DataFrame.isna : Detect missing values in a DataFrame.     Index.isna : Detect missing values in an Index.          Examples     --------     Scalar arguments (including strings) result in a scalar boolean.          >>> pd.isna('dog')     False          >>> pd.isna(pd.NA)     True          >>> pd.isna(np.nan)     True          ndarrays result in an ndarray of booleans.          >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])     >>> array     array([[ 1., nan,  3.],            [ 4.,  5., nan]])     >>> pd.isna(array)     array([[False,  True, False],            [False, False,  True]])          For indexes, an ndarray of booleans is returned.          >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,     ...                           "2017-07-08"])     >>> index     DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],                   dtype='datetime64[ns]', freq=None)     >>> pd.isna(index)     array([False, False,  True, False])          For Series and DataFrame, the same type is returned, containing booleans.          >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])     >>> df          0     1    2     0  ant   bee  cat     1  dog  None  fly     >>> pd.isna(df)            0      1      2     0  False  False  False     1  False   True  False          >>> pd.isna(df[1])     0    False     1     True     Name: 1, dtype: bool  
2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series          >>> df.apply(lambda x: [1, 2], axis=1)     0    [1, 2]     1    [1, 2]     2    [1, 2]     dtype: object          Passing ``
3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)     Remove missing values.          See the :ref:`User Guide <missing_data>` for more on which values are     considered missing, and how to work with missing data.          Parameters     ----------     axis : {0 or 'index', 1 or 'columns'}, default 0         Determine if rows or columns which contain missing values are         removed.              * 0, or 'index' : Drop rows which contain missing values.         * 1, or 'columns' : Drop columns which contain missing value.              .. versionchanged:: 1.0.0                 Pass tuple or list to drop on multiple axes.            Only a single axis is allowed.          how : {'any', 'all'}, default 'any'         Determine if row or column is removed from DataFrame, when we have         at least one NA or all NA.              * 'any' : If any NA values are present, drop that row or column.         * 'all' : If all values are NA, drop that row or column.          thresh : int, optional         Require that many non-NA values.     subset : array-like, optional         Labels along other axis to consider, e.g. if you are dropping rows         these would be a list of columns to include.     inplace : bool, default False         If True, do operation inplace and return None.          Returns     -------     DataFrame or None         DataFrame with NA entries dropped from it or None if ``inplace=True``.          See Also     --------     DataFrame.isna: Indicate missing values.     DataFrame.notna : Indicate existing (non-missing) values.     DataFrame.fillna : Replace missing values.     Series.dropna : Drop missing values.     Index.dropna : Drop missing indices.          Examples     --------     >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],     ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],     ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),     ...                             pd.NaT]})     >>> df            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Drop the rows where at least one element is missing.          >>> df.dropna()          name        toy       born     1  Batman  Batmobile 1940-04-25          Drop the columns where at least one element is missing.          >>> df.dropna(axis='columns')            name     0    Alfred     1    Batman     2  Catwoman          Drop the rows where all elements are missing.          >>> df.dropna(how='all')            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep only the rows with at least 2 non-NA values.          >>> df.dropna(thresh=2)            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Define in which columns to look for missing values.          >>> df.dropna(subset=['name', 'toy'])            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep the DataFrame with valid entries in the same variable.          >>> df.dropna(inplace=

## Unfinished Code Snippet:
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =

## Completion:
We can use function `isnull()` to detect missing values, and use `sorted` function to sort the non-null values.
To align all data, we can use `apply()` in DataFrame.
Then we can use `dropna()` in DataFrame to drop rows where all elements are NaN.
So the completion code is:
```
 df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## Potential documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)     Render a DataFrame to a console-friendly tabular output.          Parameters     ----------     buf : str, Path or StringIO-like, optional, default None         Buffer to write to. If None, the output is returned as a string.     columns : sequence, optional, default None         The subset of columns to write. Writes all columns by default.     col_space : int, list or dict of int, optional         The minimum width of each column.     header : bool or sequence, optional         Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.     index : bool, optional, default True         Whether to print index (row) labels.     na_rep : str, optional, default 'NaN'         String representation of ``NaN`` to use.     formatters : list, tuple or dict of one-param. functions, optional         Formatter functions to apply to columns' elements by position or         name.         The result of each function must be a unicode string.         List/tuple must be of length equal to the number of columns.     float_format : one-parameter function, optional, default None         Formatter function to apply to columns' elements if they are         floats. This function must return a unicode string and will be         applied only to the non-``NaN`` elements, with ``NaN`` being         handled by ``na_rep``.              .. versionchanged:: 1.2.0          sparsify : bool, optional, default True         Set to False for a DataFrame with a hierarchical index to print         every multiindex key at each row.     index_names : bool, optional, default True         Prints the names of the indexes.     justify : str, default None         How to justify the column labels. If None uses the option from         the print configuration (controlled by set_option), 'right' out         of the box. Valid values are              * left         * right         * center         * justify         * justify-all         * start         * end         * inherit         * match-parent         * initial         * unset.     max_rows : int, optional         Maximum number of rows to display in the console.     min_rows : int, optional         The number of rows to display in the console in a truncated repr         (when number of rows is above `max_rows`).     max_cols : int, optional         Maximum number of columns to display in the console.     show_dimensions : bool, default False         Display DataFrame dimensions (number of rows by number of columns).     decimal : str, default '.'         Character recognized as decimal separator, e.g. ',' in Europe.          line_width : int, optional         Width to wrap a line in characters.     max_colwidth : int, optional         Max width to truncate each column in characters. By default, no limit.              .. versionadded:: 1.0.0     encoding : str, default "utf-8"         Set character encoding.              .. versionadded:: 1.0          Returns     -------     str or None         If buf is None, returns the result as a string. Otherwise returns 

## Unfinished Code Snippet:
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =

## Completion:
To render a DataFrame to a string representation, we can use the `to_string` method of the DataFrame
So, the completion code is:
```
 df.to_string(index=False)
```
'''
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{examples_prompt}
\n
## Potential documents:
{potential_docs}
## Unfinished Code Snippet:
{question}

## Completion:
"""
    if existing_output is not None: user_prompt = user_prompt + '\n' + existing_output
    if 'gpt' in model:
        prompt = ['', user_prompt]
    else:
        prompt = user_prompt
    # prompt_template = ensemble_prompt(sys_prompt='', user_prompt=user_prompt, model=model)
    return prompt


def prompt_con(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
## Unfinished Code Snippet: 
{question}
"""
    SYS_PROMPT_CON = """follow the instruction to solve the problem.
Instruction:
1. Read the given unfinished code snippet and potential documents to gather relevant information.
2. Write reading notes summarizing the key points from these API documents.
3. Discuss the relevance of the unfinished code snippet and the documents.
4. If some documents are relevant to the given unfinished code snippet, complete the code snippet based on the documents, the completed code should be tagged with ```.
5. If no document is relevant, directly complete the code snippet without considering the documents, the completed code should be tagged with ```.
"""
    prompt = ensemble_prompt(SYS_PROMPT_CON, user_prompt, model)
    return prompt


def prompt_least_to_most(ret_docs, question, model):
    examples_prompt = '''
## Potential documents:
0: arange()     arange([start,] stop[, step,], dtype=None, *, like=None)          Return evenly spaced values within a given interval.          Values are generated within the half-open interval ``[start, stop)``     (in other words, the interval including `start` but excluding `stop`).     For integer arguments the function is equivalent to the Python built-in     `range` function, but returns an ndarray rather than a list.          When using a non-integer step, such as 0.1, the results will often not     be consistent.  It is better to use `numpy.linspace` for these cases.          Parameters     ----------     start : integer or real, optional         Start of interval.  The interval includes this value.  The default         start value is 0.     stop : integer or real         End of interval.  The interval does not include this value, except         in some cases where `step` is not an integer and floating point         round-off affects the length of `out`.     step : integer or real, optional         Spacing between values.  For any output `out`, this is the distance         between two adjacent values, ``out[i+1] - out[i]``.  The default         step size is 1.  If `step` is specified as a position argument,         `start` must also be given.     dtype : dtype         The type of the output array.  If `dtype` is not given, infer the data         type from the other input arguments.     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     arange : ndarray         Array of evenly spaced values.              For floating point arguments, the length of the result is         ``ceil((stop - start)/step)``.  Because of floating point overflow,         this rule may result in the last element of `out` being greater         than `stop`.          See Also     --------     numpy.linspace : Evenly spaced numbers with careful handling of endpoints.     numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.     numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.          Examples     --------     >>> np.arange(3)     array([0, 1, 2])     >>> np.arange(3.0)     array([ 0.,  1.,  2.])     >>> np.arange(3,7)     array([3, 4, 5, 6])     >>> np.arange(3,7,2)     array([3, 5])  

## Unfinished Code Snippet: 
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1

## Completion:
Step 1: Decompose the Problem into sub-questions:
1. How to generate indices for the sub-sequences using NumPy's broadcasting capabilities?
2. How to slice the array according to the generated indices to get the sub-sequences?
   
Step 2: Answer Each Sub-question
1. How to generate indices for the sub-sequences using NumPy's broadcasting capabilities?
   - Using `np.arange` and appropriate broadcasting operations, we can generate the required indices.
   
2. How to slice the array according to the generated indices to get the sub-sequences?
   - Using the calculated indices, we can slice the array to obtain the sub-sequences.
   
Step 3: Generate Code to complete the given code snippet
```
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## Potential documents:
0: sorted(iterable, *, key=None, reverse=False)     Return a new list containing all items from the iterable in ascending order.          A custom key function can be supplied to customize the sort order, and the     reverse flag can be set to request the result in descending order.  
1: isna(obj)     Detect missing values for an array-like object.          This function takes a scalar or array-like object and indicates     whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``     in object arrays, ``NaT`` in datetimelike).          Parameters     ----------     obj : scalar or array-like         Object to check for null or missing values.          Returns     -------     bool or array-like of bool         For scalar input, returns a scalar boolean.         For array input, returns an array of boolean indicating whether each         corresponding element is missing.          See Also     --------     notna : Boolean inverse of pandas.isna.     Series.isna : Detect missing values in a Series.     DataFrame.isna : Detect missing values in a DataFrame.     Index.isna : Detect missing values in an Index.          Examples     --------     Scalar arguments (including strings) result in a scalar boolean.          >>> pd.isna('dog')     False          >>> pd.isna(pd.NA)     True          >>> pd.isna(np.nan)     True          ndarrays result in an ndarray of booleans.          >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])     >>> array     array([[ 1., nan,  3.],            [ 4.,  5., nan]])     >>> pd.isna(array)     array([[False,  True, False],            [False, False,  True]])          For indexes, an ndarray of booleans is returned.          >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,     ...                           "2017-07-08"])     >>> index     DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],                   dtype='datetime64[ns]', freq=None)     >>> pd.isna(index)     array([False, False,  True, False])          For Series and DataFrame, the same type is returned, containing booleans.          >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])     >>> df          0     1    2     0  ant   bee  cat     1  dog  None  fly     >>> pd.isna(df)            0      1      2     0  False  False  False     1  False   True  False          >>> pd.isna(df[1])     0    False     1     True     Name: 1, dtype: bool  
2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series          >>> df.apply(lambda x: [1, 2], axis=1)     0    [1, 2]     1    [1, 2]     2    [1, 2]     dtype: object          Passing ``
3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)     Remove missing values.          See the :ref:`User Guide <missing_data>` for more on which values are     considered missing, and how to work with missing data.          Parameters     ----------     axis : {0 or 'index', 1 or 'columns'}, default 0         Determine if rows or columns which contain missing values are         removed.              * 0, or 'index' : Drop rows which contain missing values.         * 1, or 'columns' : Drop columns which contain missing value.              .. versionchanged:: 1.0.0                 Pass tuple or list to drop on multiple axes.            Only a single axis is allowed.          how : {'any', 'all'}, default 'any'         Determine if row or column is removed from DataFrame, when we have         at least one NA or all NA.              * 'any' : If any NA values are present, drop that row or column.         * 'all' : If all values are NA, drop that row or column.          thresh : int, optional         Require that many non-NA values.     subset : array-like, optional         Labels along other axis to consider, e.g. if you are dropping rows         these would be a list of columns to include.     inplace : bool, default False         If True, do operation inplace and return None.          Returns     -------     DataFrame or None         DataFrame with NA entries dropped from it or None if ``inplace=True``.          See Also     --------     DataFrame.isna: Indicate missing values.     DataFrame.notna : Indicate existing (non-missing) values.     DataFrame.fillna : Replace missing values.     Series.dropna : Drop missing values.     Index.dropna : Drop missing indices.          Examples     --------     >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],     ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],     ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),     ...                             pd.NaT]})     >>> df            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Drop the rows where at least one element is missing.          >>> df.dropna()          name        toy       born     1  Batman  Batmobile 1940-04-25          Drop the columns where at least one element is missing.          >>> df.dropna(axis='columns')            name     0    Alfred     1    Batman     2  Catwoman          Drop the rows where all elements are missing.          >>> df.dropna(how='all')            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep only the rows with at least 2 non-NA values.          >>> df.dropna(thresh=2)            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Define in which columns to look for missing values.          >>> df.dropna(subset=['name', 'toy'])            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep the DataFrame with valid entries in the same variable.          >>> df.dropna(inplace=

## Unfinished Code Snippet:
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =

## Completion:
Step 1: Decompose the Problem into sub-questions:
1. How to identify the missing values in the DataFrame?
2. How to sort non-null values within each column to bring non-null values to the top?
3. How to drop all rows where all elements are NaN?

Step 2: Answer Each Sub-question
1. How to identify the missing values in the DataFrame?
   - We can use the `pd.isnull` function from Document 1 to detect missing values.
   
2. How to sort non-null values within each column to bring non-null values to the top?
   - We can use the `sorted` function from Document 0 to sort the non-null values, combined with column-wise processing like in `DataFrame.apply` from Document 2.

3. How to drop all rows where all elements are NaN?
   - We can use the `dropna` function from Document 3 to drop rows where all elements are NaN.
   
Step 3: Generate Code to complete the given code snippet:
```
 df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## Potential documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)     Render a DataFrame to a console-friendly tabular output.          Parameters     ----------     buf : str, Path or StringIO-like, optional, default None         Buffer to write to. If None, the output is returned as a string.     columns : sequence, optional, default None         The subset of columns to write. Writes all columns by default.     col_space : int, list or dict of int, optional         The minimum width of each column.     header : bool or sequence, optional         Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.     index : bool, optional, default True         Whether to print index (row) labels.     na_rep : str, optional, default 'NaN'         String representation of ``NaN`` to use.     formatters : list, tuple or dict of one-param. functions, optional         Formatter functions to apply to columns' elements by position or         name.         The result of each function must be a unicode string.         List/tuple must be of length equal to the number of columns.     float_format : one-parameter function, optional, default None         Formatter function to apply to columns' elements if they are         floats. This function must return a unicode string and will be         applied only to the non-``NaN`` elements, with ``NaN`` being         handled by ``na_rep``.              .. versionchanged:: 1.2.0          sparsify : bool, optional, default True         Set to False for a DataFrame with a hierarchical index to print         every multiindex key at each row.     index_names : bool, optional, default True         Prints the names of the indexes.     justify : str, default None         How to justify the column labels. If None uses the option from         the print configuration (controlled by set_option), 'right' out         of the box. Valid values are              * left         * right         * center         * justify         * justify-all         * start         * end         * inherit         * match-parent         * initial         * unset.     max_rows : int, optional         Maximum number of rows to display in the console.     min_rows : int, optional         The number of rows to display in the console in a truncated repr         (when number of rows is above `max_rows`).     max_cols : int, optional         Maximum number of columns to display in the console.     show_dimensions : bool, default False         Display DataFrame dimensions (number of rows by number of columns).     decimal : str, default '.'         Character recognized as decimal separator, e.g. ',' in Europe.          line_width : int, optional         Width to wrap a line in characters.     max_colwidth : int, optional         Max width to truncate each column in characters. By default, no limit.              .. versionadded:: 1.0.0     encoding : str, default "utf-8"         Set character encoding.              .. versionadded:: 1.0          Returns     -------     str or None         If buf is None, returns the result as a string. Otherwise returns 

## Unfinished Code Snippet:
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =

## Completion:
Step 1: Decompose the Problem into sub-questions:
1. How to render a DataFrame to a string representation?

Step 2: Answer Each Sub-question:
1. How to render a DataFrame to a string representation?
   - We can use the `to_string` method of the DataFrame class to render the DataFrame as a string.

Step 3: Generate Code to complete the given code snippet:
```
 df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{examples_prompt}
\n
## Potential documents:
{potential_docs}
## Unfinished Code Snippet:
{question}

## Completion:
"""
    if 'gpt' in model:
        prompt = [SYS_PROMPT_LEAST_TO_MOST, user_prompt]
    else:
        prompt = user_prompt
    # prompt_template = ensemble_prompt(sys_prompt='', user_prompt=user_prompt, model=model)
    return prompt




def prompt_3shot(ret_docs, question, model):
    examples_3shot = '''
## Potential documents:
0: arange()     arange([start,] stop[, step,], dtype=None, *, like=None)          Return evenly spaced values within a given interval.          Values are generated within the half-open interval ``[start, stop)``     (in other words, the interval including `start` but excluding `stop`).     For integer arguments the function is equivalent to the Python built-in     `range` function, but returns an ndarray rather than a list.          When using a non-integer step, such as 0.1, the results will often not     be consistent.  It is better to use `numpy.linspace` for these cases.          Parameters     ----------     start : integer or real, optional         Start of interval.  The interval includes this value.  The default         start value is 0.     stop : integer or real         End of interval.  The interval does not include this value, except         in some cases where `step` is not an integer and floating point         round-off affects the length of `out`.     step : integer or real, optional         Spacing between values.  For any output `out`, this is the distance         between two adjacent values, ``out[i+1] - out[i]``.  The default         step size is 1.  If `step` is specified as a position argument,         `start` must also be given.     dtype : dtype         The type of the output array.  If `dtype` is not given, infer the data         type from the other input arguments.     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     arange : ndarray         Array of evenly spaced values.              For floating point arguments, the length of the result is         ``ceil((stop - start)/step)``.  Because of floating point overflow,         this rule may result in the last element of `out` being greater         than `stop`.          See Also     --------     numpy.linspace : Evenly spaced numbers with careful handling of endpoints.     numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.     numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.          Examples     --------     >>> np.arange(3)     array([0, 1, 2])     >>> np.arange(3.0)     array([ 0.,  1.,  2.])     >>> np.arange(3,7)     array([3, 4, 5, 6])     >>> np.arange(3,7,2)     array([3, 5])  

## Unfinished Code Snippet: 
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1
    
## Completion:
```
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]
```



## Potential documents:
0: sorted(iterable, *, key=None, reverse=False)     Return a new list containing all items from the iterable in ascending order.          A custom key function can be supplied to customize the sort order, and the     reverse flag can be set to request the result in descending order.  
1: isna(obj)     Detect missing values for an array-like object.          This function takes a scalar or array-like object and indicates     whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``     in object arrays, ``NaT`` in datetimelike).          Parameters     ----------     obj : scalar or array-like         Object to check for null or missing values.          Returns     -------     bool or array-like of bool         For scalar input, returns a scalar boolean.         For array input, returns an array of boolean indicating whether each         corresponding element is missing.          See Also     --------     notna : Boolean inverse of pandas.isna.     Series.isna : Detect missing values in a Series.     DataFrame.isna : Detect missing values in a DataFrame.     Index.isna : Detect missing values in an Index.          Examples     --------     Scalar arguments (including strings) result in a scalar boolean.          >>> pd.isna('dog')     False          >>> pd.isna(pd.NA)     True          >>> pd.isna(np.nan)     True          ndarrays result in an ndarray of booleans.          >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])     >>> array     array([[ 1., nan,  3.],            [ 4.,  5., nan]])     >>> pd.isna(array)     array([[False,  True, False],            [False, False,  True]])          For indexes, an ndarray of booleans is returned.          >>> index = pd.DatetimeIndex(["2017-07-05", "2017-07-06", None,     ...                           "2017-07-08"])     >>> index     DatetimeIndex(['2017-07-05', '2017-07-06', 'NaT', '2017-07-08'],                   dtype='datetime64[ns]', freq=None)     >>> pd.isna(index)     array([False, False,  True, False])          For Series and DataFrame, the same type is returned, containing booleans.          >>> df = pd.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])     >>> df          0     1    2     0  ant   bee  cat     1  dog  None  fly     >>> pd.isna(df)            0      1      2     0  False  False  False     1  False   True  False          >>> pd.isna(df[1])     0    False     1     True     Name: 1, dtype: bool  
2: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series          >>> df.apply(lambda x: [1, 2], axis=1)     0    [1, 2]     1    [1, 2]     2    [1, 2]     dtype: object          Passing ``
3: dropna(axis: 'Axis' = 0, how: 'str' = 'any', thresh=None, subset=None, inplace: 'bool' = False)     Remove missing values.          See the :ref:`User Guide <missing_data>` for more on which values are     considered missing, and how to work with missing data.          Parameters     ----------     axis : {0 or 'index', 1 or 'columns'}, default 0         Determine if rows or columns which contain missing values are         removed.              * 0, or 'index' : Drop rows which contain missing values.         * 1, or 'columns' : Drop columns which contain missing value.              .. versionchanged:: 1.0.0                 Pass tuple or list to drop on multiple axes.            Only a single axis is allowed.          how : {'any', 'all'}, default 'any'         Determine if row or column is removed from DataFrame, when we have         at least one NA or all NA.              * 'any' : If any NA values are present, drop that row or column.         * 'all' : If all values are NA, drop that row or column.          thresh : int, optional         Require that many non-NA values.     subset : array-like, optional         Labels along other axis to consider, e.g. if you are dropping rows         these would be a list of columns to include.     inplace : bool, default False         If True, do operation inplace and return None.          Returns     -------     DataFrame or None         DataFrame with NA entries dropped from it or None if ``inplace=True``.          See Also     --------     DataFrame.isna: Indicate missing values.     DataFrame.notna : Indicate existing (non-missing) values.     DataFrame.fillna : Replace missing values.     Series.dropna : Drop missing values.     Index.dropna : Drop missing indices.          Examples     --------     >>> df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],     ...                    "toy": [np.nan, 'Batmobile', 'Bullwhip'],     ...                    "born": [pd.NaT, pd.Timestamp("1940-04-25"),     ...                             pd.NaT]})     >>> df            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Drop the rows where at least one element is missing.          >>> df.dropna()          name        toy       born     1  Batman  Batmobile 1940-04-25          Drop the columns where at least one element is missing.          >>> df.dropna(axis='columns')            name     0    Alfred     1    Batman     2  Catwoman          Drop the rows where all elements are missing.          >>> df.dropna(how='all')            name        toy       born     0    Alfred        NaN        NaT     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep only the rows with at least 2 non-NA values.          >>> df.dropna(thresh=2)            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Define in which columns to look for missing values.          >>> df.dropna(subset=['name', 'toy'])            name        toy       born     1    Batman  Batmobile 1940-04-25     2  Catwoman   Bullwhip        NaT          Keep the DataFrame with valid entries in the same variable.          >>> df.dropna(inplace=

## Unfinished Code Snippet:
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =

## Completion:
```
 df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
```



## Potential documents:
0: to_string(buf: 'FilePathOrBuffer[str] | None' = None, columns: 'Sequence[str] | None' = None, col_space: 'int | None' = None, header: 'bool | Sequence[str]' = True, index: 'bool' = True, na_rep: 'str' = 'NaN', formatters: 'fmt.FormattersType | None' = None, float_format: 'fmt.FloatFormatType | None' = None, sparsify: 'bool | None' = None, index_names: 'bool' = True, justify: 'str | None' = None, max_rows: 'int | None' = None, min_rows: 'int | None' = None, max_cols: 'int | None' = None, show_dimensions: 'bool' = False, decimal: 'str' = '.', line_width: 'int | None' = None, max_colwidth: 'int | None' = None, encoding: 'str | None' = None)     Render a DataFrame to a console-friendly tabular output.          Parameters     ----------     buf : str, Path or StringIO-like, optional, default None         Buffer to write to. If None, the output is returned as a string.     columns : sequence, optional, default None         The subset of columns to write. Writes all columns by default.     col_space : int, list or dict of int, optional         The minimum width of each column.     header : bool or sequence, optional         Write out the column names. If a list of strings is given, it is assumed to be aliases for the column names.     index : bool, optional, default True         Whether to print index (row) labels.     na_rep : str, optional, default 'NaN'         String representation of ``NaN`` to use.     formatters : list, tuple or dict of one-param. functions, optional         Formatter functions to apply to columns' elements by position or         name.         The result of each function must be a unicode string.         List/tuple must be of length equal to the number of columns.     float_format : one-parameter function, optional, default None         Formatter function to apply to columns' elements if they are         floats. This function must return a unicode string and will be         applied only to the non-``NaN`` elements, with ``NaN`` being         handled by ``na_rep``.              .. versionchanged:: 1.2.0          sparsify : bool, optional, default True         Set to False for a DataFrame with a hierarchical index to print         every multiindex key at each row.     index_names : bool, optional, default True         Prints the names of the indexes.     justify : str, default None         How to justify the column labels. If None uses the option from         the print configuration (controlled by set_option), 'right' out         of the box. Valid values are              * left         * right         * center         * justify         * justify-all         * start         * end         * inherit         * match-parent         * initial         * unset.     max_rows : int, optional         Maximum number of rows to display in the console.     min_rows : int, optional         The number of rows to display in the console in a truncated repr         (when number of rows is above `max_rows`).     max_cols : int, optional         Maximum number of columns to display in the console.     show_dimensions : bool, default False         Display DataFrame dimensions (number of rows by number of columns).     decimal : str, default '.'         Character recognized as decimal separator, e.g. ',' in Europe.          line_width : int, optional         Width to wrap a line in characters.     max_colwidth : int, optional         Max width to truncate each column in characters. By default, no limit.              .. versionadded:: 1.0.0     encoding : str, default "utf-8"         Set character encoding.              .. versionadded:: 1.0          Returns     -------     str or None         If buf is None, returns the result as a string. Otherwise returns 

## Unfinished Code Snippet:
df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =

## Completion:
```
 df.to_string(index=False)
```
'''

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
{examples_3shot}
\n
## Potential documents:
{potential_docs}
## Unfinished Code Snippet:
{question}

## Completion:
"""
    if 'gpt' in model: prompt = ['', user_prompt]
    else: prompt = user_prompt
    # prompt_template = ensemble_prompt(sys_prompt='', user_prompt=user_prompt, model=model)
    return prompt


def prompt_0shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Unfinished Code Snippet:
{question}
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_0shot_no_ret(question, model, pads=''):
    user_prompt = f"""
{pads}\n
## Unfinished Code Snippet:
{question}
"""
    sys_prompt = LLAMA_SYSTEM_PROMPT_NO_RET
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_plan_and_solve(ret_docs, question, model):
    plan_and_solve_prompt = """You are a senior Python programmer, given some Potential Documents and a Uncompleted Code Snippet,
Your task is to first understand the Uncompleted Code Snippet and devise a plan to complete it.
Then, you should carry out the plan to complete the code snippet"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Uncompleted Code Snippet:
{question}
"""
    sys_prompt = plan_and_solve_prompt
    prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt


def prompt_RaR(ret_docs, question, model):
    RaR_prompt = """You are a senior python programmer, given some Potential Documents and a Uncompleted Code Snippet
Your task is to first rephrase and expand the Problem, then complete the code snippet"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Uncompleted Code Snippet:
{question}
"""
    sys_prompt = RaR_prompt
    prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt



if __name__ == '__main__':
    """get random samples as few shot examples"""
    import sys, platform
    import random
    import json

    system = platform.system()
    if system == 'Darwin':
        root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    sys.path.insert(0, root_path)
    from dataset_utils.pandas_numpy_eval_utils import PandasNumpyEvalLoader
    from dataset_utils.corpus_utils import PythonDocsLoader
    from generator.generate_utils import truncate_docs


    def get_few_shots(k):
        loader = PandasNumpyEvalLoader()
        qs_list = loader.load_qs_list()
        qs_id_list = [qs['qs_id'] for qs in qs_list]
        dataset = json.load(open(loader.data_file))
        unincluded_dataset = []
        for data in dataset:
            if data['task_id'] not in qs_id_list:
                unincluded_dataset.append(data)
        random.seed()
        few_shots = random.sample(unincluded_dataset, k=k)
        for shot in few_shots:
            print(shot['prompt'])
            print(shot['canonical_solution'])
            print(shot.keys())


    # get_few_shots(k=3)


    '''
import numpy as np

def broadcasting_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    """
    I want to create a matrix of sub sequences from this array of length L with stride S.
    Return the numpy array of sub sequences.
    """
    nrows = ((a.size-L)//S)+1

['    return a[S*np.arange(nrows)[:,None] + np.arange(L)]']

    '''

    '''
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': [1, 4, 7, np.nan], 'B': [np.nan, 2, 5, np.nan], 'C': [np.nan, np.nan, 3, 6]})
# Move next value to first empty row pandas
# how do i move each value from a column to the first empty "row/cell" in pandas?
# use sorted to align non NULL data at the top, use dropna to drop all rows with all NaN
new_df =
[" df.apply(lambda x: sorted(x, key=pd.isnull)).dropna(how = 'all')"]
    '''

    '''
    df = pd.DataFrame({'a': [0, 1], 'b': [5, 3]})
# How to obtain pandas DataFrame without index
# I want to print the whole dataframe, but I don't want to print the index
df_string =
[' df.to_string(index=False)']
    '''

    api_signs = [['matplotlib.pylab.arange'], ["builtins.sorted", "pandas.core.common.isna", "pandas.core.frame.DataFrame.apply", "pandas.core.frame.DataFrame.dropna"], ["pandas.core.frame.DataFrame.to_string"]]
    for api_sign in api_signs:
        docs = PythonDocsLoader().get_docs(api_sign)
        docs = truncate_docs(docs, model='codellama-13b-instruct', max_length=1000)
        potential_docs = ''
        for idx, ret_doc in enumerate(docs):
            potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
        print(potential_docs)
