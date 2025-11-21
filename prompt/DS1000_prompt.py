import sys
sys.path.append('/home/zhaoshengming/RAG_Benchmark_Code_QA')
from prompt.prompt_utils import ensemble_prompt


# LLAMA_SYSTEM_PROMPT = """You are a senior python programmer.
#
# Input:
# - Some potentially useful api documents tagged `## Potential documents`
# - A program description tagged `## Problem`
# - Incomplete code tagged `## Incomplete code`.
#
# Task:
# Complete the code by replacing `[insert]` with the correct Python code.
#
# Output Rules:
# 1. Only change `[insert]` to working Python code, keep everything else exactly the same
# 2. Output the complete code in <code> and </code> tags
# """

LLAMA_SYSTEM_PROMPT = """You are a senior python programmer. 

Input:
- Useful api documents tagged `## API Documents`
- A program description tagged `## Problem`
- Incomplete code tagged `## Incomplete code`.

Task:
Follow the API documents and the problem description, to complete the code by replacing `[insert]` with the correct Python code.

Output Rules:
1. Only change `[insert]` to working Python code, keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""

# todo: new NO RET prompt
LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer.

Input:
- A program description tagged `## Problem`
- Incomplete code tagged `## Incomplete code`.

Task:
Complete the code by replacing `[insert]` with the correct Python code.

Output Rules:
1. Only change `[insert]` to working Python code, keep everything else exactly the same
2. Output the complete code in <code> and </code> tags
"""


SYS_PROMPT_ZERO_SHOT = """Input:
- Useful api documents tagged `## API Documents`
- A program description tagged `## Problem`
- Incomplete code tagged `## Incomplete code`.

Task:
Follow the API documents and the problem description, to complete the code by replacing `[insert]` with the correct Python code.

Output Rules:
1. Only change `[insert]` to working Python code, keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""



# # todo: OG No Ret prompt
# LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer, given a program description tagged `## Problem` and the incomplete code tagged `## Incomplete Code`, your task is to complete the code by replacing `[insert]` with the correct Python code.
# You should generate the complete code solution without changing the existing code, and the output code should in <code> and </code> tags
# """

# SYS_PROMPT_LEAST_TO_MOST = """Follow the examples to solve the last problem"""


def prompt_0shot(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>', '').replace('</code>', '').replace('BEGIN SOLUTION', '').replace('END SOLUTION', '')
    user_prompt = f"""
## API Documents:
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
#     if model.startswith('llama2') or model.startswith('codellama'):
#         prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{user_prompt} [/INST]"""
#     elif model.startswith('llama3'):
#         prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>
# """
#     elif model.startswith('gpt'):
#         prompt_template = sys_prompt + '\n' + user_prompt
#     else:
#         raise ValueError(f'Unrecognized model: {model}')
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_0shot_no_ret(question, model, pads=''):
    prompt, answer = question.split('\nA:')
    prompt = prompt.replace('Problem:', '')
    answer = answer.replace('<code>', '').replace('</code>', '').replace('BEGIN SOLUTION', '').replace('END SOLUTION', '')
    user_prompt = f"""
{pads}\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
"""
    prompt_template = ensemble_prompt(sys_prompt=LLAMA_SYSTEM_PROMPT_NO_RET, user_prompt=user_prompt, model=model)
    return prompt_template


def process_docs_question(ret_docs, question):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    prompt, answer = question.split('\nA:')
    prompt = prompt.replace('Problem:', '')
    return potential_docs, prompt, answer




def prompt_3shot(ret_docs, question, model):

    examples_prompt = """## API Documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.
    
    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.
    
    .. warning:: The default `atol` is not appropriate for comparing numbers
                 that are much smaller than one (see Notes).
    
    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.
    
    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.
    
    See Also
    --------
    allclose
    math.isclose
    
    Notes
    -----
    .. versionadded:: 1.7.0
    
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.
    
     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    
    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
    the default value of atol is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity: if the expected values
    are significantly smaller than one, it can result in false positives.
    `atol` should be carefully selected for the use case at hand. A
    
    
1: sum(iterable, start=0)
    Return the sum of a 'start' value (default: 0) plus an iterable of numbers
    
    When the iterable is empty, return the start value.
    This function is intended specifically for use with numeric values and may
    reject non-numeric types.



## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed).
Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values.
Here is a simple standalone example to illustrate this issue :
import numpy as np
n = 10
m = 4
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
# print the number of times s1 is not equal to s2 (should be 0)
print np.nonzero(s1 != s2)[0].shape[0]
If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance.
The problem is I need to use those in functions like np.in1d where I can't really give a tolerance...
What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above.
Is there a way to avoid this issue?


## Incomplete Code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
[insert]
print(result)
```


## Complete code by replacing [insert] with the correct Python code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
result = (~np.isclose(s1,s2)).sum()
print(result)
```



## API Documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)
    Gather slices from `params` into a Tensor with shape specified by `indices`.
    
    `indices` is a `Tensor` of indices into `params`. The index vectors are
    arranged along the last axis of `indices`.
    
    This is similar to `tf.gather`, in which `indices` defines slices into the
    first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the
    first `N` dimensions of `params`, where `N = indices.shape[-1]`.
    
    Caution: On CPU, if an out of bound index is found, an error is returned.
    On GPU, if an out of bound index is found, a 0 is stored in the
    corresponding output value.
    
    ## Gathering scalars
    
    In the simplest case the vectors in `indices` index the full rank of `params`:
    
    >>> tf.gather_nd(
    ...     indices=[[0, 0],
    ...              [1, 1]],
    ...     params = [['a', 'b'],
    ...               ['c', 'd']]).numpy()
    array([b'a', b'd'], dtype=object)
    
    In this case the result has 1-axis fewer than `indices`, and each index vector
    is replaced by the scalar indexed from `params`.
    
    In this case the shape relationship is:
    
    ```
    index_depth = indices.shape[-1]
    assert index_depth == params.shape.rank
    result_shape = indices.shape[:-1]
    ```
    
    If `indices` has a rank of `K`, it is helpful to think `indices` as a
    (K-1)-dimensional tensor of indices into `params`.
    
    ## Gathering slices
    
    If the index vectors do not index the full rank of `params` then each location
    in the result contains a slice of params. This example collects rows from a
    matrix:
    
    >>> tf.gather_nd(
    ...     indices = [[1],
    ...                [0]],
    ...     params = [['a', 'b', 'c'],
    ...               ['d', 'e', 'f']]).numpy()
    array([[b'd', b'e', b'f'],
           [b'a', b'b', b'c']], dtype
           
           
## Problem: 
I'm using tensorflow 2.10.0.

import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
m = x[y,z]

What I expect is m = [2,6]
I can get the result by theano or numpy. How I get the result using tensorflow?


## Incomplete Code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
[insert]
print(result)
```



## Complete code by replacing [insert] with the correct Python code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
print(result)
```



## API Documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)
    Join columns of another DataFrame.
    
    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.
    
    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.
    
        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.
    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).
    
    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.
    
    See Also
    --------


1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
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
    

2: add_prefix(prefix: 'str')
    Prefix labels with string `prefix`.
    
    For Series, the row labels are prefixed.
    For DataFrame, the column labels are prefixed.
    
    Parameters
    ----------
    prefix : str
        The string to add before each label.
    
    Returns
    -------
    Series or DataFrame
        New Series or DataFrame with updated labels.
    
    See Also
    --------
    Series.add_suffix: Suffix row labels with string `suffix`.
    DataFrame.add_suffix: Suffix column labels with string `suffix`.
    
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    
    >>> s.add_prefix('item_')
    item_0    1
    item_1    2
    item_2    3
    item_3    4
    dtype: int64
    
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
    >>> df
       A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6
    
    >>> df.add_prefix('col_')
         col_A  col_B
    0       1       3
    1       2       4
    2       3       5
    3       4       6



## Problem: 
Sample dataframe:
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on.
The resulting dataframe should look like so:
result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})

Notice that e is the natural constant.
Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.


## Incomplete code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
[insert]
print(result)
```



## Complete code by replacing [insert] with the correct Python code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())
print(result)
```
"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')
    answer = f'```{answer}\n```'
    user_prompt = f"""{examples_prompt}
\n\n
## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
\n
## Complete code by replacing [insert] with the correct Python code:
"""

    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt


if __name__ == '__main__':
    example_doc = "add_prefix(prefix: 'str')\n    Prefix labels with string `prefix`.\n    \n    For Series, the row labels are prefixed.\n    For DataFrame, the column labels are prefixed.\n    \n    Parameters\n    ----------\n    prefix : str\n        The string to add before each label.\n    \n    Returns\n    -------\n    Series or DataFrame\n        New Series or DataFrame with updated labels.\n    \n    See Also\n    --------\n    Series.add_suffix: Suffix row labels with string `suffix`.\n    DataFrame.add_suffix: Suffix column labels with string `suffix`.\n    \n    Examples\n    --------\n    >>> s = pd.Series([1, 2, 3, 4])\n    >>> s\n    0    1\n    1    2\n    2    3\n    3    4\n    dtype: int64\n    \n    >>> s.add_prefix('item_')\n    item_0    1\n    item_1    2\n    item_2    3\n    item_3    4\n    dtype: int64\n    \n    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})\n    >>> df\n       A  B\n    0  1  3\n    1  2  4\n    2  3  5\n    3  4  6\n    \n    >>> df.add_prefix('col_')\n         col_A  col_B\n    0       1       3\n    1       2       4\n    2       3       5\n    3       4       6\n\n"

    from generator_deprecated.generate_utils import truncate_docs
    print(truncate_docs([example_doc], 'gpt-3.5-turbo-0125', 500)[0])





def prompt_emotion(ret_docs, question, model):
    system_prompt_emotion = """This is very important to my career.
    
Input:
- Useful api documents tagged `## API Documents`
- A program description tagged `## Problem`
- Incomplete code tagged `## Incomplete code`.

Task:
Follow the API documents and the problem description, to complete the code by replacing `[insert]` with the correct Python code.

Output Rules:
1. Only change `[insert]` to working Python code, keep existing code exactly the same
2. Output the complete code in <code> and </code> tags
"""
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
"""
    prompt_template = ensemble_prompt(sys_prompt=system_prompt_emotion, user_prompt=user_prompt, model=model)
    return prompt_template





def prompt_cot(ret_docs, question, model, existing_output=None):
    cot_prompt = """## API Documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.
    
    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.
    
    .. warning:: The default `atol` is not appropriate for comparing numbers
                 that are much smaller than one (see Notes).
    
    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.
    
    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.
    
    See Also
    --------
    allclose
    math.isclose
    
    Notes
    -----
    .. versionadded:: 1.7.0
    
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.
    
     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))
    
    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
    the default value of atol is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity: if the expected values
    are significantly smaller than one, it can result in false positives.
    `atol` should be carefully selected for the use case at hand. A
    
    
1: sum(iterable, start=0)
    Return the sum of a 'start' value (default: 0) plus an iterable of numbers
    
    When the iterable is empty, return the start value.
    This function is intended specifically for use with numeric values and may
    reject non-numeric types.



## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed).
Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values.
Here is a simple standalone example to illustrate this issue :
import numpy as np
n = 10
m = 4
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
# print the number of times s1 is not equal to s2 (should be 0)
print np.nonzero(s1 != s2)[0].shape[0]
If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance.
The problem is I need to use those in functions like np.in1d where I can't really give a tolerance...
What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above.
Is there a way to avoid this issue?


## Incomplete Code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
[insert]
print(result)
```


According to the problem, I need to count genuinely different elements while ignoring floating-point precision errors.
The document shows isclose() for tolerance-based comparison and sum() for counting boolean results.
I should use negated isclose() to find truly different elements, then sum the boolean array.
Based on this, I implement the complete code by replacing `[insert]` with the correct Python code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
result = (~np.isclose(s1,s2)).sum()
print(result)
```



## API Documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)
    Gather slices from `params` into a Tensor with shape specified by `indices`.
    
    `indices` is a `Tensor` of indices into `params`. The index vectors are
    arranged along the last axis of `indices`.
    
    This is similar to `tf.gather`, in which `indices` defines slices into the
    first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the
    first `N` dimensions of `params`, where `N = indices.shape[-1]`.
    
    Caution: On CPU, if an out of bound index is found, an error is returned.
    On GPU, if an out of bound index is found, a 0 is stored in the
    corresponding output value.
    
    ## Gathering scalars
    
    In the simplest case the vectors in `indices` index the full rank of `params`:
    
    >>> tf.gather_nd(
    ...     indices=[[0, 0],
    ...              [1, 1]],
    ...     params = [['a', 'b'],
    ...               ['c', 'd']]).numpy()
    array([b'a', b'd'], dtype=object)
    
    In this case the result has 1-axis fewer than `indices`, and each index vector
    is replaced by the scalar indexed from `params`.
    
    In this case the shape relationship is:
    
    ```
    index_depth = indices.shape[-1]
    assert index_depth == params.shape.rank
    result_shape = indices.shape[:-1]
    ```
    
    If `indices` has a rank of `K`, it is helpful to think `indices` as a
    (K-1)-dimensional tensor of indices into `params`.
    
    ## Gathering slices
    
    If the index vectors do not index the full rank of `params` then each location
    in the result contains a slice of params. This example collects rows from a
    matrix:
    
    >>> tf.gather_nd(
    ...     indices = [[1],
    ...                [0]],
    ...     params = [['a', 'b', 'c'],
    ...               ['d', 'e', 'f']]).numpy()
    array([[b'd', b'e', b'f'],
           [b'a', b'b', b'c']], dtype
           
           
## Problem: 
I'm using tensorflow 2.10.0.

import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
m = x[y,z]

What I expect is m = [2,6]
I can get the result by theano or numpy. How I get the result using tensorflow?


## Incomplete Code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
[insert]
print(result)
```


According to the problem, I need to index a 2D tensor using coordinate pairs from two 1D tensors.
The document shows gather_nd() which extracts elements using indices arranged along the last axis.
Since I need coordinate pairs, I should stack y and z into coordinate pairs format [[0,1], [1,2]] for gather_nd.
Based on this, I implement the complete code by replacing `[insert]` with the correct Python code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
print(result)
```



## API Documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)
    Join columns of another DataFrame.
    
    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.
    
    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.
    
        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.
    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).
    
    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.
    
    See Also
    --------


1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
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
    

2: add_prefix(prefix: 'str')
    Prefix labels with string `prefix`.
    
    For Series, the row labels are prefixed.
    For DataFrame, the column labels are prefixed.
    
    Parameters
    ----------
    prefix : str
        The string to add before each label.
    
    Returns
    -------
    Series or DataFrame
        New Series or DataFrame with updated labels.
    
    See Also
    --------
    Series.add_suffix: Suffix row labels with string `suffix`.
    DataFrame.add_suffix: Suffix column labels with string `suffix`.
    
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64
    
    >>> s.add_prefix('item_')
    item_0    1
    item_1    2
    item_2    3
    item_3    4
    dtype: int64
    
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
    >>> df
       A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6
    
    >>> df.add_prefix('col_')
         col_A  col_B
    0       1       3
    1       2       4
    2       3       5
    3       4       6



## Problem: 
Sample dataframe:
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on.
The resulting dataframe should look like so:
result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})

Notice that e is the natural constant.
Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.


## Incomplete code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
[insert]
print(result)
```


According to the problem, I need to add sigmoid columns with prefixed names to the existing dataframe.
The document shows apply() for applying functions to columns, add_prefix() for renaming columns, and join() for combining DataFrames.
I should apply sigmoid to all columns, add the prefix, then join with the original dataframe.
Based on this, I implement the complete code by replacing `[insert]` with the correct Python code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())
print(result)
```
"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')
    answer = f'```{answer}\n```'
    user_prompt = f"""{cot_prompt}
\n\n
## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
"""
    if existing_output is not None: user_prompt = user_prompt + '\n' + existing_output
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt


def prompt_zero_shot_cot(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}

Let's think it step by step.
"""
    prompt_template = ensemble_prompt(sys_prompt=SYS_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template







def prompt_con(ret_docs, question, model):
    system_prompt_con = """Input:
- Useful api documents tagged `## API Documents`
- A program description tagged `## Problem`
- Incomplete code tagged `## Incomplete code`.

Task:
Follow the API documents and the problem description to complete the code by:
1. Reading the problem and API documents to gather relevant information
2. Writing brief reading notes summarizing key points from the API documents
3. Assessing the relevance between the user's problem and available API functions
4. Using relevant API functions to complete the python code, or implementing without the given APIs if none are relevant

Output Format:
Reading notes: [Brief summary of key API functions and their purposes]
Relevance assessment: [How the APIs relate to the problem]
```
[Complete python code, only replace `[insert]` with the correct Python code.]
```
"""
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
"""
    prompt_template = ensemble_prompt(sys_prompt=system_prompt_con, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_self_refine(ret_docs, question, model, initial_output):
    examples_prompt = """## API Documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 that are much smaller than one (see Notes).

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose
    math.isclose

    Notes
    -----
    .. versionadded:: 1.7.0

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
    the default value of atol is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity: if the expected values
    are significantly smaller than one, it can result in false positives.
    `atol` should be carefully selected for the use case at hand. A


1: sum(iterable, start=0)
    Return the sum of a 'start' value (default: 0) plus an iterable of numbers

    When the iterable is empty, return the start value.
    This function is intended specifically for use with numeric values and may
    reject non-numeric types.



## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed).
Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values.
Here is a simple standalone example to illustrate this issue :
import numpy as np
n = 10
m = 4
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
# print the number of times s1 is not equal to s2 (should be 0)
print np.nonzero(s1 != s2)[0].shape[0]
If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance.
The problem is I need to use those in functions like np.in1d where I can't really give a tolerance...
What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above.
Is there a way to avoid this issue?


## Incomplete Code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
[insert]
print(result)
```


## Initial Code Solution:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
result = (s1 != s2).sum()
print(result)
```

Please first provide feedback on this solution and then refine it based on the feedback.


Feedback: The current solution uses direct inequality comparison (s1 != s2), which will count floating-point precision errors as genuine differences. 
This doesn't solve the core problem - we want to ignore tiny numerical differences and only count truly different elements.

Refined solution:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
result = (~np.isclose(s1,s2)).sum()
print(result)
```




## API Documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)
    Gather slices from `params` into a Tensor with shape specified by `indices`.

    `indices` is a `Tensor` of indices into `params`. The index vectors are
    arranged along the last axis of `indices`.

    This is similar to `tf.gather`, in which `indices` defines slices into the
    first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the
    first `N` dimensions of `params`, where `N = indices.shape[-1]`.

    Caution: On CPU, if an out of bound index is found, an error is returned.
    On GPU, if an out of bound index is found, a 0 is stored in the
    corresponding output value.

    ## Gathering scalars

    In the simplest case the vectors in `indices` index the full rank of `params`:

    >>> tf.gather_nd(
    ...     indices=[[0, 0],
    ...              [1, 1]],
    ...     params = [['a', 'b'],
    ...               ['c', 'd']]).numpy()
    array([b'a', b'd'], dtype=object)

    In this case the result has 1-axis fewer than `indices`, and each index vector
    is replaced by the scalar indexed from `params`.

    In this case the shape relationship is:

    ```
    index_depth = indices.shape[-1]
    assert index_depth == params.shape.rank
    result_shape = indices.shape[:-1]
    ```

    If `indices` has a rank of `K`, it is helpful to think `indices` as a
    (K-1)-dimensional tensor of indices into `params`.

    ## Gathering slices

    If the index vectors do not index the full rank of `params` then each location
    in the result contains a slice of params. This example collects rows from a
    matrix:

    >>> tf.gather_nd(
    ...     indices = [[1],
    ...                [0]],
    ...     params = [['a', 'b', 'c'],
    ...               ['d', 'e', 'f']]).numpy()
    array([[b'd', b'e', b'f'],
           [b'a', b'b', b'c']], dtype


## Problem: 
I'm using tensorflow 2.10.0.

import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
m = x[y,z]

What I expect is m = [2,6]
I can get the result by theano or numpy. How I get the result using tensorflow?


## Incomplete Code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
[insert]
print(result)
```


## Initial Code Solution:
```
import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
result = tf.gather(tf.gather(x, y), z)
print(result)
```


Please first provide feedback on this solution and then refine it based on the feedback.


Feedback: The current solution uses nested tf.gather() calls, which doesn't correctly implement coordinate-based indexing.
This approach would gather entire rows first, then try to gather from those rows, but it doesn't properly handle the coordinate pairs (y[i], z[i]).
We need to use gather_nd() with properly formatted coordinate indices.

Refined solution:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
print(result)
```



## API Documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)
    Join columns of another DataFrame.

    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.

    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.

        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.
    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).

    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.

    See Also
    --------


1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
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


2: add_prefix(prefix: 'str')
    Prefix labels with string `prefix`.

    For Series, the row labels are prefixed.
    For DataFrame, the column labels are prefixed.

    Parameters
    ----------
    prefix : str
        The string to add before each label.

    Returns
    -------
    Series or DataFrame
        New Series or DataFrame with updated labels.

    See Also
    --------
    Series.add_suffix: Suffix row labels with string `suffix`.
    DataFrame.add_suffix: Suffix column labels with string `suffix`.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.add_prefix('item_')
    item_0    1
    item_1    2
    item_2    3
    item_3    4
    dtype: int64

    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
    >>> df
       A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6

    >>> df.add_prefix('col_')
         col_A  col_B
    0       1       3
    1       2       4
    2       3       5
    3       4       6



## Problem: 
Sample dataframe:
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on.
The resulting dataframe should look like so:
result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})

Notice that e is the natural constant.
Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.


## Incomplete code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
[insert]
print(result)
```


## Initial Code Solution:
```
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
import math
sigmoid_df = df.apply(lambda x: 1/(1+math.e**(-x)))
sigmoid_df.columns = ['sigmoid_' + col for col in sigmoid_df.columns]
result = pd.concat([df, sigmoid_df], axis=1)
print(result)
```


Please first provide feedback on this solution and then refine it based on the feedback.



Feedback: The current solution manually renames columns using list comprehension and uses concat() to combine dataframes. 
While this works, it's not leveraging pandas' built-in methods efficiently. 
We can use add_prefix() for cleaner column renaming and join() for more direct combination.

Refined solution:
```
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
import math
result = df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))
print(result)
```
"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')
    answer = f"```{answer}```"
    user_prompt = f"""{examples_prompt}
\n\n
## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}
\n
# Initial Code Solution:
```
{initial_output}
```

Please first provide feedback on this solution and then refine it based on the feedback.
"""


    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt




def prompt_least_to_most(ret_docs, question, model):
    examples_prompt = """## API Documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    .. warning:: The default `atol` is not appropriate for comparing numbers
                 that are much smaller than one (see Notes).

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    See Also
    --------
    allclose
    math.isclose

    Notes
    -----
    .. versionadded:: 1.7.0

    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    Unlike the built-in `math.isclose`, the above equation is not symmetric
    in `a` and `b` -- it assumes `b` is the reference value -- so that
    `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,
    the default value of atol is not zero, and is used to determine what
    small values should be considered close to zero. The default value is
    appropriate for expected values of order unity: if the expected values
    are significantly smaller than one, it can result in false positives.
    `atol` should be carefully selected for the use case at hand. A


1: sum(iterable, start=0)
    Return the sum of a 'start' value (default: 0) plus an iterable of numbers

    When the iterable is empty, return the start value.
    This function is intended specifically for use with numeric values and may
    reject non-numeric types.



## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed).
Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values.
Here is a simple standalone example to illustrate this issue :
import numpy as np
n = 10
m = 4
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
# print the number of times s1 is not equal to s2 (should be 0)
print np.nonzero(s1 != s2)[0].shape[0]
If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance.
The problem is I need to use those in functions like np.in1d where I can't really give a tolerance...
What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above.
Is there a way to avoid this issue?


## Incomplete Code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
[insert]
print(result)
```



Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: What's causing the comparison issue?
Floating-point arithmetic creates tiny precision differences when summing in different orders, making s1 != s2 even when they should be equal.

Sub problem 2: How to compare arrays with tolerance for precision errors?
I can use isclose() to check if elements are approximately equal within a tolerance, accounting for floating-point precision.

Sub problem 3: How to count elements that are truly different?
I should negate isclose() with ~ to find genuinely different elements, then sum() to count them.

Now combining all solutions to generate complete code, by replacing [insert] with the correct Python code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
result = (~np.isclose(s1,s2)).sum()
print(result)
```




## API Documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)
    Gather slices from `params` into a Tensor with shape specified by `indices`.

    `indices` is a `Tensor` of indices into `params`. The index vectors are
    arranged along the last axis of `indices`.

    This is similar to `tf.gather`, in which `indices` defines slices into the
    first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the
    first `N` dimensions of `params`, where `N = indices.shape[-1]`.

    Caution: On CPU, if an out of bound index is found, an error is returned.
    On GPU, if an out of bound index is found, a 0 is stored in the
    corresponding output value.

    ## Gathering scalars

    In the simplest case the vectors in `indices` index the full rank of `params`:

    >>> tf.gather_nd(
    ...     indices=[[0, 0],
    ...              [1, 1]],
    ...     params = [['a', 'b'],
    ...               ['c', 'd']]).numpy()
    array([b'a', b'd'], dtype=object)

    In this case the result has 1-axis fewer than `indices`, and each index vector
    is replaced by the scalar indexed from `params`.

    In this case the shape relationship is:

    ```
    index_depth = indices.shape[-1]
    assert index_depth == params.shape.rank
    result_shape = indices.shape[:-1]
    ```

    If `indices` has a rank of `K`, it is helpful to think `indices` as a
    (K-1)-dimensional tensor of indices into `params`.

    ## Gathering slices

    If the index vectors do not index the full rank of `params` then each location
    in the result contains a slice of params. This example collects rows from a
    matrix:

    >>> tf.gather_nd(
    ...     indices = [[1],
    ...                [0]],
    ...     params = [['a', 'b', 'c'],
    ...               ['d', 'e', 'f']]).numpy()
    array([[b'd', b'e', b'f'],
           [b'a', b'b', b'c']], dtype


## Problem: 
I'm using tensorflow 2.10.0.

import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
m = x[y,z]

What I expect is m = [2,6]
I can get the result by theano or numpy. How I get the result using tensorflow?


## Incomplete Code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
[insert]
print(result)
```



Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: What indexing operation do I need?
I want to get x[0,1]=2 and x[1,2]=6, which means indexing with coordinate pairs (y[i], z[i]).

Sub problem 2: How to create coordinate pairs for TensorFlow?
I need to stack y=[0,1] and z=[1,2] to form [[0,1], [1,2]] using tf.stack.

Sub problem 3: Which TensorFlow function can perform coordinate-based indexing?
I can use gather_nd() to extract elements using the coordinate pairs.

Now combining all solutions to generate complete code, by replacing [insert] with the correct Python code:
```
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
print(result)
```



## API Documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)
    Join columns of another DataFrame.

    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.

    Parameters
    ----------
    other : DataFrame, Series, or list of DataFrame
        Index should be similar to one of the columns in this one. If a
        Series is passed, its name attribute must be set, and that will be
        used as the column name in the resulting joined DataFrame.
    on : str, list of str, or array-like, optional
        Column or index level name(s) in the caller to join on the index
        in `other`, otherwise joins index-on-index. If multiple
        values given, the `other` DataFrame must have a MultiIndex. Can
        pass an array as the join key if it is not already contained in
        the calling DataFrame. Like an Excel VLOOKUP operation.
    how : {'left', 'right', 'outer', 'inner'}, default 'left'
        How to handle the operation of the two objects.

        * left: use calling frame's index (or column if on is specified)
        * right: use `other`'s index.
        * outer: form union of calling frame's index (or column if on is
          specified) with `other`'s index, and sort it.
          lexicographically.
        * inner: form intersection of calling frame's index (or column if
          on is specified) with `other`'s index, preserving the order
          of the calling's one.
    lsuffix : str, default ''
        Suffix to use from left frame's overlapping columns.
    rsuffix : str, default ''
        Suffix to use from right frame's overlapping columns.
    sort : bool, default False
        Order result DataFrame lexicographically by the join key. If False,
        the order of the join key depends on the join type (how keyword).

    Returns
    -------
    DataFrame
        A dataframe containing columns from both the caller and `other`.

    See Also
    --------


1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)
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


2: add_prefix(prefix: 'str')
    Prefix labels with string `prefix`.

    For Series, the row labels are prefixed.
    For DataFrame, the column labels are prefixed.

    Parameters
    ----------
    prefix : str
        The string to add before each label.

    Returns
    -------
    Series or DataFrame
        New Series or DataFrame with updated labels.

    See Also
    --------
    Series.add_suffix: Suffix row labels with string `suffix`.
    DataFrame.add_suffix: Suffix column labels with string `suffix`.

    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])
    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.add_prefix('item_')
    item_0    1
    item_1    2
    item_2    3
    item_3    4
    dtype: int64

    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
    >>> df
       A  B
    0  1  3
    1  2  4
    2  3  5
    3  4  6

    >>> df.add_prefix('col_')
         col_A  col_B
    0       1       3
    1       2       4
    2       3       5
    3       4       6



## Problem: 
Sample dataframe:
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on.
The resulting dataframe should look like so:
result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})

Notice that e is the natural constant.
Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.


## Incomplete code:
```
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
[insert]
print(result)
```



Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: How to apply sigmoid function to all columns?
I can use apply() with a lambda function to compute sigmoid: 1/(1+math.e**(-x)) for each column.

Sub problem 2: How to add prefix to the resulting column names?
I can use add_prefix('sigmoid_') to rename columns from 'A', 'B' to 'sigmoid_A', 'sigmoid_B'.

Sub problem 3: How to combine the original and sigmoid DataFrames?
I can use join() to merge the original dataframe with the prefixed sigmoid dataframe.

Now combining all solutions to generate complete code, by replacing [insert] with the correct Python code:
```
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))
result = g(df.copy())
print(result)
```
"""


    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')
    answer = f'```{answer}\n```'
    user_prompt = f"""
{examples_prompt}
\n\n
## API Documents:
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}

Let me break this down into simpler sub problems and solve them step by step:
"""
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt



def prompt_plan_and_solve(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    answer = answer.replace('<code>\n', '').replace('</code>\n', '').replace('BEGIN SOLUTION\n', '').replace('END SOLUTION\n', '')

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Incomplete Code:
{answer}

Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step.
"""
    prompt_template = ensemble_prompt(sys_prompt=SYS_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template




# def prompt_RaR(ret_docs, question, model):
#     RaR_prompt = """You are a senior python programmer, given some potential documents, a problem and its unfinished code solution
# Your task is to first rephrase and expand the Problem, then complete the Unfinished Code Solution without modifying its existing part"""
#
#     potential_docs, prompt, answer = process_docs_question(ret_docs, question)
#     user_prompt = f"""
# ## Potential documents:
# {potential_docs}
# ## Problem:
# {prompt}
#
# ## Unfinished Code Solution:
# {answer}
# """
#     sys_prompt = RaR_prompt
#     prompt = ensemble_prompt(sys_prompt, user_prompt, model)
#     # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
#     return prompt



if __name__ == '__main__':
    """
    get examples
    """
    import sys, platform
    import random

    system = platform.system()
    if system == 'Darwin':
        root_path = '/'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/RAG_Benchmark_Code_QA'
    sys.path.insert(0, root_path)
    from dataset_utils.DS1000_utils import DS1000Loader
    from generator_deprecated.generate_utils import truncate_docs
    from dataset_utils.corpus_utils import PythonDocsLoader
    from data.DS1000.ds1000 import DS1000Dataset, DS1000Problem
    ds1000 = DS1000Dataset(source_dir=root_path + '/data/DS1000/ds1000_data', mode='Insertion', libs='all')

    # def classify_prompt(data_list):
    #     type_1_list, type_2_list = [], []
    #     for data in data_list:
    #         if 'Problem:' in data['nl']:
    #             type_1_list.append(data)
    #         else:
    #             type_2_list.append(data)
    #     return type_1_list, type_2_list
    #
    # ds1000_loader = DS1000Loader()
    # sampled_data = ds1000_loader.load_qs_list(sampled=True)
    # sampled_data_type1, sampled_data_type2 = classify_prompt(sampled_data)
    # sampled_id_type1, sampled_id_type2 = [data['qs_id'] for data in sampled_data_type1], [data['qs_id'] for data in sampled_data_type2]
    # all_data = ds1000_loader.load_qs_list(sampled=False)
    # all_data_type1, all_data_type2 = classify_prompt(all_data)
    # all_id_type1, all_id_type2 = [data['qs_id'] for data in all_data_type1], [data['qs_id'] for data in all_data_type2]
    # rest_id_type1, rest_id_type2 = [id for id in all_id_type1 if id not in sampled_id_type1], [id for id in all_id_type2 if id not in sampled_id_type2]
    # sampled_rest_id_type1 = random.sample(rest_id_type1, 10)
    # sampled_rest_id_type2 = random.sample(rest_id_type2, 10)
    #
    # print(sampled_rest_id_type1)
    # print(sampled_rest_id_type2)


    sampled_rest_id_type1 = ['Numpy_200', 'Tensorflow_25', 'Pandas_53', 'Numpy_41', 'Scipy_98', 'Scipy_66', 'Tensorflow_8', 'Numpy_96', 'Scipy_56', 'Numpy_166']
    sampled_rest_id_type2 = ['Matplotlib_93', 'Matplotlib_142', 'Matplotlib_145', 'Matplotlib_33', 'Matplotlib_80', 'Matplotlib_20', 'Matplotlib_44', 'Matplotlib_120', 'Matplotlib_14', 'Matplotlib_99']


    """
    show shots
    """
    def get_sampled_prompt(shots=3):
        for sampled_id in sampled_rest_id_type1[:shots]:
            [lib, problem_id] = sampled_id.split('_')
            data = ds1000[lib][int(problem_id)]
            print('qs_id', sampled_id)
            print('reference_code', data['reference_code'])
            prompt, code = data['prompt'].split('A:')
            print('prompt:\n', prompt.replace('\n', ' '))
            print('code:\n', code)
        for sampled_id in sampled_rest_id_type2[:shots]:
            [lib, problem_id] = sampled_id.split('_')
            data = ds1000[lib][int(problem_id)]
            print('qs_id', sampled_id)
            print('reference_code', data['reference_code'])
            print('code:\n', data['prompt'])

    # get_sampled_prompt(shots=3)


    """
    ensemble shots
    """
    def ensemble_shots(qs_ids, api_signs):
        python_loader = PythonDocsLoader()
        examples, answers = [], []
        for qs_id, api_sign in zip(qs_ids, api_signs):
            [lib, problem_id] = qs_id.split('_')
            data = ds1000[lib][int(problem_id)]
            docs = python_loader.get_docs(api_sign)
            potential_docs, prompt, answer = process_docs_question(docs, data['prompt'])
            if answer is not None:
                example = f"""## Potential documents:
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Unfinished Code Solution:
{answer}
\n
"""
            else:
                example = f"""## Potential documents:
{potential_docs}
\n
## Unfinished Code Snippet:
{prompt}
\n
"""
            answer = f"""
{data['reference_code']}
"""
            examples.append(example)
            answers.append(answer)
        return examples, answers


    # type1
    # qs_ids = ['Numpy_200', 'Tensorflow_25', 'Pandas_53']
    # api_signs = [['numpy.isclose', 'builtins.sum'], ['tensorflow.gather_nd'], ['pandas.core.frame.DataFrame.join', 'pandas.core.frame.DataFrame.apply', 'pandas.core.frame.DataFrame.add_prefix']]
    # examples, answers = ensemble_shots(qs_ids, api_signs)
    # for example, answer in zip(examples, answers):
    #     print(example)
    #     print(answer)

    # type2
    # qs_ids = ['Matplotlib_93', 'Matplotlib_142', 'Matplotlib_145']
    # api_signs = [['matplotlib.pyplot.yticks'], ['matplotlib.pyplot.plot', 'matplotlib.pyplot.tick_params'], ['seaborn.categorical.catplot', 'numpy.ndarray.flatten', 'matplotlib.axes._axes.Axes.set_ylabel']]
    # examples, answers = ensemble_shots(qs_ids, api_signs)
    # for example, answer in zip(examples, answers):
    #     print(example)
    #     print(answer)


