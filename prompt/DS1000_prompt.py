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

# # todo: OG No Ret prompt
# LLAMA_SYSTEM_PROMPT_NO_RET = """You are a senior python programmer, given a program description tagged `## Problem` and the incomplete code tagged `## Incomplete Code`, your task is to complete the code by replacing `[insert]` with the correct Python code.
# You should generate the complete code solution without changing the existing code, and the output code should in <code> and </code> tags
# """

SYS_PROMPT_LEAST_TO_MOST = """Follow the examples to solve the last problem"""


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


# def llama_3shots_prompt(ret_docs, question, model):
#     if '\nA:' in question:
#         return llama_3shots_prompt_type1(ret_docs, question, model)
#     else:
#         return llama_3shots_prompt_type2(ret_docs, question, model)


cot_prompt = """
## Potential documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)     Returns a boolean array where two arrays are element-wise equal within a     tolerance.          The tolerance values are positive, typically very small numbers.  The     relative difference (`rtol` * abs(`b`)) and the absolute difference     `atol` are added together to compare against the absolute difference     between `a` and `b`.          .. warning:: The default `atol` is not appropriate for comparing numbers                  that are much smaller than one (see Notes).          Parameters     ----------     a, b : array_like         Input arrays to compare.     rtol : float         The relative tolerance parameter (see Notes).     atol : float         The absolute tolerance parameter (see Notes).     equal_nan : bool         Whether to compare NaN's as equal.  If True, NaN's in `a` will be         considered equal to NaN's in `b` in the output array.          Returns     -------     y : array_like         Returns a boolean array of where `a` and `b` are equal within the         given tolerance. If both `a` and `b` are scalars, returns a single         boolean value.          See Also     --------     allclose     math.isclose          Notes     -----     .. versionadded:: 1.7.0          For finite values, isclose uses the following equation to test whether     two floating point values are equivalent.           absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))          Unlike the built-in `math.isclose`, the above equation is not symmetric     in `a` and `b` -- it assumes `b` is the reference value -- so that     `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,     the default value of atol is not zero, and is used to determine what     small values should be considered close to zero. The default value is     appropriate for expected values of order unity: if the expected values     are significantly smaller than one, it can result in false positives.     `atol` should be carefully selected for the use case at hand. A zero value     for `atol` will result in `False` if either `a` or `b` is zero.          `isclose` is not defined for non-numeric data types.          Examples     --------     >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])     array([ True, False])     >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])     array([ True, True])     >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])     array([False,  True])     >>> np.isclose([1.0, np.nan], [1.0, np.nan])     array([ True, False])     >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)     array([ True, True])     >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])     array([ True, False])     >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)     array([False, False])     >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])     array([ True,  True])
1: sum(iterable, start=0)     Return the sum of a 'start' value (default: 0) plus an iterable of numbers          When the iterable is empty, return the start value.     This function is intended specifically for use with numeric values and may     reject non-numeric types.  

## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed). Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values. Here is a simple standalone example to illustrate this issue : import numpy as np n = 10 m = 4 tag = np.random.rand(n, m) s1 = np.sum(tag, axis=1) s2 = np.sum(tag[:, ::-1], axis=1) # print the number of times s1 is not equal to s2 (should be 0) print np.nonzero(s1 != s2)[0].shape[0] If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance. The problem is I need to use those in functions like np.in1d where I can't really give a tolerance... What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above. Is there a way to avoid this issue? 

## Unfinished Code Solution:
<code>
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
We first use `numpy.isclose()` to compare `s1` and `s2` element-wise with in a tolerance
Then, we use sum() to count the number of elements that are equal
So the code in [insert] is:
```
result = (~np.isclose(s1,s2)).sum()
```



## Potential documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)     Gather slices from `params` into a Tensor with shape specified by `indices`.          `indices` is a `Tensor` of indices into `params`. The index vectors are     arranged along the last axis of `indices`.          This is similar to `tf.gather`, in which `indices` defines slices into the     first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the     first `N` dimensions of `params`, where `N = indices.shape[-1]`.          Caution: On CPU, if an out of bound index is found, an error is returned.     On GPU, if an out of bound index is found, a 0 is stored in the     corresponding output value.          ## Gathering scalars          In the simplest case the vectors in `indices` index the full rank of `params`:          >>> tf.gather_nd(     ...     indices=[[0, 0],     ...              [1, 1]],     ...     params = [['a', 'b'],     ...               ['c', 'd']]).numpy()     array([b'a', b'd'], dtype=object)          In this case the result has 1-axis fewer than `indices`, and each index vector     is replaced by the scalar indexed from `params`.          In this case the shape relationship is:          ```     index_depth = indices.shape[-1]     assert index_depth == params.shape.rank     result_shape = indices.shape[:-1]     ```          If `indices` has a rank of `K`, it is helpful to think `indices` as a     (K-1)-dimensional tensor of indices into `params`.          ## Gathering slices          If the index vectors do not index the full rank of `params` then each location     in the result contains a slice of params. This example collects rows from a     matrix:          >>> tf.gather_nd(     ...     indices = [[1],     ...                [0]],     ...     params = [['a', 'b', 'c'],     ...               ['d', 'e', 'f']]).numpy()     array([[b'd', b'e', b'f'],            [b'a', b'b', b'c']], dtype=object)          Here `indices` contains `[2]` index vectors, each with a length of `1`.     The index vectors each refer to rows of the `params` matrix. Each     row has a shape of `[3]` so the output shape is `[2, 3]`.          In this case, the relationship between the shapes is:          ```     index_depth = indices.shape[-1]     outer_shape = indices.shape[:-1]     assert index_depth <= params.shape.rank     inner_shape = params.shape[index_depth:]     output_shape = outer_shape + inner_shape     ```          It is helpful to think of the results in this case as tensors-of-tensors.     The shape of the outer tensor is set by the leading dimensions of `indices`.     While the shape of the inner tensors is the shape of a single slice.          ## Batches          Additionally both `params` and `indices` can have `M` leading batch     dimensions that exactly match. In this case `batch_dims` must be set to `M`.          For example, to collect one row from each of a batch of matrices you could     set the leading elements of the index vectors to be their location in the     batch:          >>> tf.gather_nd(     ...     indices = [[0, 1],     ...                [1, 0],     ...                [2, 4],     ...                [3, 2],     ...                [4, 1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          The `batch_dims` argument lets you omit those leading location dimensions     from the index:          >>> tf.gather_nd(     ...     batch_dims=1,     ...     indices = [[1],     ...                [0],     ...                [4],     ...                [2],     ...                [1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          This is equivalent to caling a separate `gather_nd` for each location in the     batch dimensions.               >>> params=tf.zeros([5, 7, 3])     >>> indices=tf.zeros([5, 1])     >>> batch_dims = 1     >>>     >>> index_depth = indices.shape[-1]     >>> batch_shape = indices.shape[:batch_dims]

## Problem: 
I'm using tensorflow 2.10.0.  import tensorflow as tf x = [[1,2,3],[4,5,6]] y = [0,1] z = [1,2] x = tf.constant(x) y = tf.constant(y) z = tf.constant(z) m = x[y,z]  What I expect is m = [2,6] I can get the result by theano or numpy. How I get the result using tensorflow?   

## Unfinished Code Solution:
<code>
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
we use can `tf.gather_nd` to gather specific elements from a tensor according to the indices provided
The tensor is `x` and the indices is `[y, z]`
So the code in [insert] is:
```
indices = tf.stack([y, z], axis=1)
result = tf.gather_nd(x, indices)
```



## Potential documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)     Join columns of another DataFrame.          Join columns with `other` DataFrame either on index or on a key     column. Efficiently join multiple DataFrame objects by index at once by     passing a list.          Parameters     ----------     other : DataFrame, Series, or list of DataFrame         Index should be similar to one of the columns in this one. If a         Series is passed, its name attribute must be set, and that will be         used as the column name in the resulting joined DataFrame.     on : str, list of str, or array-like, optional         Column or index level name(s) in the caller to join on the index         in `other`, otherwise joins index-on-index. If multiple         values given, the `other` DataFrame must have a MultiIndex. Can         pass an array as the join key if it is not already contained in         the calling DataFrame. Like an Excel VLOOKUP operation.     how : {'left', 'right', 'outer', 'inner'}, default 'left'         How to handle the operation of the two objects.              * left: use calling frame's index (or column if on is specified)         * right: use `other`'s index.         * outer: form union of calling frame's index (or column if on is           specified) with `other`'s index, and sort it.           lexicographically.         * inner: form intersection of calling frame's index (or column if           on is specified) with `other`'s index, preserving the order           of the calling's one.     lsuffix : str, default ''         Suffix to use from left frame's overlapping columns.     rsuffix : str, default ''         Suffix to use from right frame's overlapping columns.     sort : bool, default False         Order result DataFrame lexicographically by the join key. If False,         the order of the join key depends on the join type (how keyword).          Returns     -------     DataFrame         A dataframe containing columns from both the caller and `other`.          See Also     --------     DataFrame.merge : For column(s)-on-column(s) operations.          Notes     -----     Parameters `on`, `lsuffix`, and `rsuffix` are not supported when     passing a list of `DataFrame` objects.          Support for specifying index levels as the `on` parameter was added     in version 0.23.0.          Examples     --------     >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],     ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})          >>> df       key   A     0  K0  A0     1  K1  A1     2  K2  A2     3  K3  A3     4  K4  A4     5  K5  A5          >>> other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],     ...                       'B': ['B0', 'B1', 'B2']})          >>> other       key   B     0  K0  B0     1  K1  B1     2  K2  B2          Join DataFrames using their indexes. 
1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series
2: add_prefix(prefix: 'str')     Prefix labels with string `prefix`.          For Series, the row labels are prefixed.     For DataFrame, the column labels are prefixed.          Parameters     ----------     prefix : str         The string to add before each label.          Returns     -------     Series or DataFrame         New Series or DataFrame with updated labels.          See Also     --------     Series.add_suffix: Suffix row labels with string `suffix`.     DataFrame.add_suffix: Suffix column labels with string `suffix`.          Examples     --------     >>> s = pd.Series([1, 2, 3, 4])     >>> s     0    1     1    2     2    3     3    4     dtype: int64          >>> s.add_prefix('item_')     item_0    1     item_1    2     item_2    3     item_3    4     dtype: int64          >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})     >>> df        A  B     0  1  3     1  2  4     2  3  5     3  4  6          >>> df.add_prefix('col_')          col_A  col_B     0       1       3     1       2       4     2       3       5     3       4       6 

## Problem: 
Sample dataframe: df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on. The resulting dataframe should look like so: result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})  Notice that e is the natural constant. Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.  

## Unfinished Code Solution:
<code>
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
We first implement sigmoid function as `1/(1+math.e**(-x))`.
Then, we use function `df.apply` to create the new column with the corresponding sigmoid values
Then, we can use function `app_prefix()` to create the new column name with the desired prefix.
And we finally use function `join()` in DataFrame to to combine original dataframe with the new sigmoid columns
So, the code in [insert] is:
```
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())
```
"""


def prompt_cot(ret_docs, question, model, existing_output=None):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
{cot_prompt}
\n
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
# Code Generation for [insert]:
"""
    if existing_output is not None: user_prompt = user_prompt + '\n' + existing_output
    sys_prompt = LLAMA_SYSTEM_PROMPT
    # prompt_template = ensemble_prompt(sys_prompt, user_prompt, model, examples=[example1, example2, example3], answers=[answer1, answer2, answer3])
    if 'gpt' in model:
        prompt = ['', user_prompt]
    else:
        prompt = user_prompt
    return prompt


def prompt_con(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
"""
    SYS_PROMPT_CON = """follow the instruction to solve the problem.
Instruction:
1. Read the given problem and potential documents to gather relevant information.
2. Write reading notes summarizing the key points from these API documents.
3. Discuss the relevance of the given problem and documents.
4. If some documents are relevant to the given problem, complete the [insert] in unfinished code snippet based on the problem and the documents, the code should be tagged with ```.
5. If no document is relevant, directly complete the [insert] in unfinished code snippet without considering the documents, the code should be tagged with ```.
6. When completing the code snippet, you should not change the existing code in it
"""
    prompt = ensemble_prompt(SYS_PROMPT_CON, user_prompt, model)
    return prompt


def prompt_self_refine(ret_docs, question, model, initial_output):

    examples_prompt = """## Potential documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)     Returns a boolean array where two arrays are element-wise equal within a     tolerance.          The tolerance values are positive, typically very small numbers.  The     relative difference (`rtol` * abs(`b`)) and the absolute difference     `atol` are added together to compare against the absolute difference     between `a` and `b`.          .. warning:: The default `atol` is not appropriate for comparing numbers                  that are much smaller than one (see Notes).          Parameters     ----------     a, b : array_like         Input arrays to compare.     rtol : float         The relative tolerance parameter (see Notes).     atol : float         The absolute tolerance parameter (see Notes).     equal_nan : bool         Whether to compare NaN's as equal.  If True, NaN's in `a` will be         considered equal to NaN's in `b` in the output array.          Returns     -------     y : array_like         Returns a boolean array of where `a` and `b` are equal within the         given tolerance. If both `a` and `b` are scalars, returns a single         boolean value.          See Also     --------     allclose     math.isclose          Notes     -----     .. versionadded:: 1.7.0          For finite values, isclose uses the following equation to test whether     two floating point values are equivalent.           absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))          Unlike the built-in `math.isclose`, the above equation is not symmetric     in `a` and `b` -- it assumes `b` is the reference value -- so that     `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,     the default value of atol is not zero, and is used to determine what     small values should be considered close to zero. The default value is     appropriate for expected values of order unity: if the expected values     are significantly smaller than one, it can result in false positives.     `atol` should be carefully selected for the use case at hand. A zero value     for `atol` will result in `False` if either `a` or `b` is zero.          `isclose` is not defined for non-numeric data types. 
1: sum(iterable, start=0)     Return the sum of a 'start' value (default: 0) plus an iterable of numbers          When the iterable is empty, return the start value.     This function is intended specifically for use with numeric values and may     reject non-numeric types.  

## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed). Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values. Here is a simple standalone example to illustrate this issue : import numpy as np n = 10 m = 4 tag = np.random.rand(n, m) s1 = np.sum(tag, axis=1) s2 = np.sum(tag[:, ::-1], axis=1) # print the number of times s1 is not equal to s2 (should be 0) print np.nonzero(s1 != s2)[0].shape[0] If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance. The problem is I need to use those in functions like np.in1d where I can't really give a tolerance... What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above. Is there a way to avoid this issue? 

## Unfinished Code Solution:
<code>
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Generated Code:
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)

# [insert]
result = (s1 != s2).sum()

print(result)
```


## Feedback and refine: Does the generated code meet the problem requirements and produce the desired results? If not, identify the issues and rewrite the code to ensure it functions correctly and outputs the expected results.
No, The code above is incorrect because it directly compares the arrays using `s1 != s2`, which does not account for floating-point precision errors.
This will cause elements that are mathematically equal but differ slightly due to precision issues to be marked as different.
We need to use `numpy.isclose()` to compare the arrays with a tolerance to handle precision issues, and we must refine it to a one-liner solution.

Let's rewrite this using the correct `numpy.isclose()` function
```
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)

# [insert]
result = (~np.isclose(s1, s2)).sum()

print(result)
```

*** END ***



## Potential documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)     Gather slices from `params` into a Tensor with shape specified by `indices`.          `indices` is a `Tensor` of indices into `params`. The index vectors are     arranged along the last axis of `indices`.          This is similar to `tf.gather`, in which `indices` defines slices into the     first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the     first `N` dimensions of `params`, where `N = indices.shape[-1]`.          Caution: On CPU, if an out of bound index is found, an error is returned.     On GPU, if an out of bound index is found, a 0 is stored in the     corresponding output value.          ## Gathering scalars          In the simplest case the vectors in `indices` index the full rank of `params`:          >>> tf.gather_nd(     ...     indices=[[0, 0],     ...              [1, 1]],     ...     params = [['a', 'b'],     ...               ['c', 'd']]).numpy()     array([b'a', b'd'], dtype=object)          In this case the result has 1-axis fewer than `indices`, and each index vector     is replaced by the scalar indexed from `params`.          In this case the shape relationship is:          ```     index_depth = indices.shape[-1]     assert index_depth == params.shape.rank     result_shape = indices.shape[:-1]     ```          If `indices` has a rank of `K`, it is helpful to think `indices` as a     (K-1)-dimensional tensor of indices into `params`.          ## Gathering slices          If the index vectors do not index the full rank of `params` then each location     in the result contains a slice of params. This example collects rows from a     matrix:          >>> tf.gather_nd(     ...     indices = [[1],     ...                [0]],     ...     params = [['a', 'b', 'c'],     ...               ['d', 'e', 'f']]).numpy()     array([[b'd', b'e', b'f'],            [b'a', b'b', b'c']], dtype=object)          Here `indices` contains `[2]` index vectors, each with a length of `1`.     The index vectors each refer to rows of the `params` matrix. Each     row has a shape of `[3]` so the output shape is `[2, 3]`. 

## Problem: 
I'm using tensorflow 2.10.0.  import tensorflow as tf x = [[1,2,3],[4,5,6]] y = [0,1] z = [1,2] x = tf.constant(x) y = tf.constant(y) z = tf.constant(z) m = x[y,z]  What I expect is m = [2,6] I can get the result by theano or numpy. How I get the result using tensorflow?   

## Unfinished Code Solution:
<code>
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Generated Code:
```
import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)

# [insert]
result = x[y, z]

print(result)
```


## Feedback and refine: Does the generated code meet the problem requirements and produce the desired results? If not, identify the issues and rewrite the code to ensure it functions correctly and outputs the expected results.
No, the code above is incorrect because direct indexing like `x[y, z]` is not supported in TensorFlow when dealing with tensors.
Instead, we need to use `tf.gather_nd()` to gather elements from `x` based on the index arrays `y` and `z`.
We need to stack y and z along the correct axis to form the necessary indexing structure.

Let's rewrite this using the correct `tf.gather_nd()` function
```
import tensorflow as tf
x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)

# [insert]
reference code def g(x,y,z):
    return tf.gather_nd(x, [y, z])
result = g(x.__copy__(),y.__copy__(),z.__copy__())

print(result)
```

*** END ***



## Potential documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)     Join columns of another DataFrame.          Join columns with `other` DataFrame either on index or on a key     column. Efficiently join multiple DataFrame objects by index at once by     passing a list.          Parameters     ----------     other : DataFrame, Series, or list of DataFrame         Index should be similar to one of the columns in this one. If a         Series is passed, its name attribute must be set, and that will be         used as the column name in the resulting joined DataFrame.     on : str, list of str, or array-like, optional         Column or index level name(s) in the caller to join on the index         in `other`, otherwise joins index-on-index. If multiple         values given, the `other` DataFrame must have a MultiIndex. Can         pass an array as the join key if it is not already contained in         the calling DataFrame. Like an Excel VLOOKUP operation.     how : {'left', 'right', 'outer', 'inner'}, default 'left'         How to handle the operation of the two objects.              * left: use calling frame's index (or column if on is specified)         * right: use `other`'s index.         * outer: form union of calling frame's index (or column if on is           specified) with `other`'s index, and sort it.           lexicographically.         * inner: form intersection of calling frame's index (or column if           on is specified) with `other`'s index, preserving the order           of the calling's one.     lsuffix : str, default ''         Suffix to use from left frame's overlapping columns.     rsuffix : str, default ''         Suffix to use from right frame's overlapping columns.     sort : bool, default False         Order result DataFrame lexicographically by the join key. If False,         the order of the join key depends on the join type (how keyword).          Returns     -------     DataFrame         A dataframe containing columns from both the caller and `other`.          See Also     --------     DataFrame.merge : For column(s)-on-column(s) operations.          Notes     -----     Parameters `on`, `lsuffix`, and `rsuffix` are not supported when     passing a list of `DataFrame` objects.
1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          
2: add_prefix(prefix: 'str')     Prefix labels with string `prefix`.          For Series, the row labels are prefixed.     For DataFrame, the column labels are prefixed.          Parameters     ----------     prefix : str         The string to add before each label.          Returns     -------     Series or DataFrame         New Series or DataFrame with updated labels.          See Also     --------     Series.add_suffix: Suffix row labels with string `suffix`.     DataFrame.add_suffix: Suffix column labels with string `suffix`.

## Problem: 
Sample dataframe: df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on. The resulting dataframe should look like so: result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})  Notice that e is the natural constant. Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.  

## Unfinished Code Solution:
<code>
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Generated Code:
```
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# [insert]
result = df.apply(lambda x: 1/(1+pd.np.exp(-x)))

print(result)
```


## Feedback and refine: Does the generated code meet the problem requirements and produce the desired results? If not, identify the issues and rewrite the code to ensure it functions correctly and outputs the expected results.
No, The initial code applies the sigmoid transformation but fails to add the correct prefixes to the new columns. 
Additionally, it uses pd.np.exp() which is deprecated; numpy should be imported directly to handle the exponential function. 
To resolve this, we need to prefix the new columns correctly and use numpy for the sigmoid operation. We will also ensure that the new columns are added to the original DataFrame.

Let's rewrite the code
```
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# [insert]
import numpy as np
def g(df):
    return df.join(df.apply(lambda x: 1 / (1 + np.exp(-x))).add_prefix('sigmoid_'))
result = g(df.copy())

print(result)
```

*** END ***
"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
{examples_prompt}
\n
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
# Generated Code:
{initial_output}


## Feedback and refine: Does the generated code meet the problem requirements and produce the desired results? If not, identify the issues and rewrite the code to ensure it functions correctly and outputs the expected results.
"""


    sys_prompt = LLAMA_SYSTEM_PROMPT
    # prompt_template = ensemble_prompt(sys_prompt, user_prompt, model, examples=[example1, example2, example3], answers=[answer1, answer2, answer3])
    if 'gpt' in model:
        prompt = ['', user_prompt]
    else:
        prompt = user_prompt
    return prompt


def prompt_3shot(ret_docs, question, model):

    examples_prompt = """
## Potential documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)     Returns a boolean array where two arrays are element-wise equal within a     tolerance.          The tolerance values are positive, typically very small numbers.  The     relative difference (`rtol` * abs(`b`)) and the absolute difference     `atol` are added together to compare against the absolute difference     between `a` and `b`.          .. warning:: The default `atol` is not appropriate for comparing numbers                  that are much smaller than one (see Notes).          Parameters     ----------     a, b : array_like         Input arrays to compare.     rtol : float         The relative tolerance parameter (see Notes).     atol : float         The absolute tolerance parameter (see Notes).     equal_nan : bool         Whether to compare NaN's as equal.  If True, NaN's in `a` will be         considered equal to NaN's in `b` in the output array.          Returns     -------     y : array_like         Returns a boolean array of where `a` and `b` are equal within the         given tolerance. If both `a` and `b` are scalars, returns a single         boolean value.          See Also     --------     allclose     math.isclose          Notes     -----     .. versionadded:: 1.7.0          For finite values, isclose uses the following equation to test whether     two floating point values are equivalent.           absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))          Unlike the built-in `math.isclose`, the above equation is not symmetric     in `a` and `b` -- it assumes `b` is the reference value -- so that     `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,     the default value of atol is not zero, and is used to determine what     small values should be considered close to zero. The default value is     appropriate for expected values of order unity: if the expected values     are significantly smaller than one, it can result in false positives.     `atol` should be carefully selected for the use case at hand. A zero value     for `atol` will result in `False` if either `a` or `b` is zero.          `isclose` is not defined for non-numeric data types.          Examples     --------     >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])     array([ True, False])     >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])     array([ True, True])     >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])     array([False,  True])     >>> np.isclose([1.0, np.nan], [1.0, np.nan])     array([ True, False])     >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)     array([ True, True])     >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])     array([ True, False])     >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)     array([False, False])     >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])     array([ True,  True])     >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)     array([False,  True])  
1: sum(iterable, start=0)     Return the sum of a 'start' value (default: 0) plus an iterable of numbers          When the iterable is empty, return the start value.     This function is intended specifically for use with numeric values and may     reject non-numeric types.  

## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed). Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values. Here is a simple standalone example to illustrate this issue : import numpy as np n = 10 m = 4 tag = np.random.rand(n, m) s1 = np.sum(tag, axis=1) s2 = np.sum(tag[:, ::-1], axis=1) # print the number of times s1 is not equal to s2 (should be 0) print np.nonzero(s1 != s2)[0].shape[0] If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance. The problem is I need to use those in functions like np.in1d where I can't really give a tolerance... What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above. Is there a way to avoid this issue? 

## Unfinished Code Solution:
<code>
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## [insert]:
```
result = (~np.isclose(s1,s2)).sum()
```



## Potential documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)     Gather slices from `params` into a Tensor with shape specified by `indices`.          `indices` is a `Tensor` of indices into `params`. The index vectors are     arranged along the last axis of `indices`.          This is similar to `tf.gather`, in which `indices` defines slices into the     first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the     first `N` dimensions of `params`, where `N = indices.shape[-1]`.          Caution: On CPU, if an out of bound index is found, an error is returned.     On GPU, if an out of bound index is found, a 0 is stored in the     corresponding output value.          ## Gathering scalars          In the simplest case the vectors in `indices` index the full rank of `params`:          >>> tf.gather_nd(     ...     indices=[[0, 0],     ...              [1, 1]],     ...     params = [['a', 'b'],     ...               ['c', 'd']]).numpy()     array([b'a', b'd'], dtype=object)          In this case the result has 1-axis fewer than `indices`, and each index vector     is replaced by the scalar indexed from `params`.          In this case the shape relationship is:          ```     index_depth = indices.shape[-1]     assert index_depth == params.shape.rank     result_shape = indices.shape[:-1]     ```          If `indices` has a rank of `K`, it is helpful to think `indices` as a     (K-1)-dimensional tensor of indices into `params`.          ## Gathering slices          If the index vectors do not index the full rank of `params` then each location     in the result contains a slice of params. This example collects rows from a     matrix:          >>> tf.gather_nd(     ...     indices = [[1],     ...                [0]],     ...     params = [['a', 'b', 'c'],     ...               ['d', 'e', 'f']]).numpy()     array([[b'd', b'e', b'f'],            [b'a', b'b', b'c']], dtype=object)          Here `indices` contains `[2]` index vectors, each with a length of `1`.     The index vectors each refer to rows of the `params` matrix. Each     row has a shape of `[3]` so the output shape is `[2, 3]`.          In this case, the relationship between the shapes is:          ```     index_depth = indices.shape[-1]     outer_shape = indices.shape[:-1]     assert index_depth <= params.shape.rank     inner_shape = params.shape[index_depth:]     output_shape = outer_shape + inner_shape     ```          It is helpful to think of the results in this case as tensors-of-tensors.     The shape of the outer tensor is set by the leading dimensions of `indices`.     While the shape of the inner tensors is the shape of a single slice.          ## Batches          Additionally both `params` and `indices` can have `M` leading batch     dimensions that exactly match. In this case `batch_dims` must be set to `M`.          For example, to collect one row from each of a batch of matrices you could     set the leading elements of the index vectors to be their location in the     batch:          >>> tf.gather_nd(     ...     indices = [[0, 1],     ...                [1, 0],     ...                [2, 4],     ...                [3, 2],     ...                [4, 1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          The `batch_dims` argument lets you omit those leading location dimensions     from the index:          >>> tf.gather_nd(     ...     batch_dims=1,     ...     indices = [[1],     ...                [0],     ...                [4],     ...                [2],     ...                [1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          This is equivalent to caling a separate `gather_nd` for each location in the     batch dimensions.               >>> params=tf.zeros([5, 7, 3])     >>> indices=tf.zeros([5, 1])     >>> batch_dims = 1     >>>     >>> index_depth = indices.shape[-1]     >>> batch_shape = indices.shape[:batch_dims]     >>> assert params.shape[:batch_dims] == batch_shape     >>> outer_shape = indices.shape[batch_dims:-1]

## Problem: 
I'm using tensorflow 2.10.0.  import tensorflow as tf x = [[1,2,3],[4,5,6]] y = [0,1] z = [1,2] x = tf.constant(x) y = tf.constant(y) z = tf.constant(z) m = x[y,z]  What I expect is m = [2,6] I can get the result by theano or numpy. How I get the result using tensorflow?   

## Unfinished Code Solution:
<code>
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## [insert]:
```
reference code def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
```



## Potential documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)     Join columns of another DataFrame.          Join columns with `other` DataFrame either on index or on a key     column. Efficiently join multiple DataFrame objects by index at once by     passing a list.          Parameters     ----------     other : DataFrame, Series, or list of DataFrame         Index should be similar to one of the columns in this one. If a         Series is passed, its name attribute must be set, and that will be         used as the column name in the resulting joined DataFrame.     on : str, list of str, or array-like, optional         Column or index level name(s) in the caller to join on the index         in `other`, otherwise joins index-on-index. If multiple         values given, the `other` DataFrame must have a MultiIndex. Can         pass an array as the join key if it is not already contained in         the calling DataFrame. Like an Excel VLOOKUP operation.     how : {'left', 'right', 'outer', 'inner'}, default 'left'         How to handle the operation of the two objects.              * left: use calling frame's index (or column if on is specified)         * right: use `other`'s index.         * outer: form union of calling frame's index (or column if on is           specified) with `other`'s index, and sort it.           lexicographically.         * inner: form intersection of calling frame's index (or column if           on is specified) with `other`'s index, preserving the order           of the calling's one.     lsuffix : str, default ''         Suffix to use from left frame's overlapping columns.     rsuffix : str, default ''         Suffix to use from right frame's overlapping columns.     sort : bool, default False         Order result DataFrame lexicographically by the join key. If False,         the order of the join key depends on the join type (how keyword).          Returns     -------     DataFrame         A dataframe containing columns from both the caller and `other`.          See Also     --------     DataFrame.merge : For column(s)-on-column(s) operations.          Notes     -----     Parameters `on`, `lsuffix`, and `rsuffix` are not supported when     passing a list of `DataFrame` objects.          Support for specifying index levels as the `on` parameter was added     in version 0.23.0.          Examples     --------     >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],     ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})          >>> df       key   A     0  K0  A0     1  K1  A1     2  K2  A2     3  K3  A3     4  K4  A4     5  K5  A5          >>> other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],     ...                       'B': ['B0', 'B1', 'B2']})          >>> other       key   B     0  K0  B0     1  K1  B1     2  K2  B2          Join DataFrames using their indexes.          >>> df.join(other, lsuffix='_caller', rsuffix='_other')       key_caller   A key_other    B     0         K0  A0        K0   B0     1         K1  A1        K1   B1     2         K2  A2        K2   B2     3         K3  A3       NaN  NaN     4         K4  A4       NaN  NaN     5         K5  A5       NaN  NaN          If we want to join using the key columns, we need to set key to be     the index in both `df` and `other`. The joined DataFrame will have     key as its index.          >>> df.set_index('key').join(other.set_index('key'))           A    B     key     K0   A0   B0     K1   A1   B1     K2   A2   B2     K3   A3  NaN     K4   A4  NaN     K5   A5  NaN          Another option to join using the key columns is to use the `on`     parameter. DataFrame
1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series          >>> df.apply(lambda x: [1, 2], axis=1)     0    [1, 2]     1    [1, 2]     2    [1, 2]     dtype: object          Passing ``result_type='expand'`` will expand list-like results     to columns of a Dataframe          >>> df.apply(lambda x: [1, 2], axis=1, result_type='expand')        0  1     0  1  2     1  1  2     2  1  2          Returning a Series inside the function is similar to passing     ``result_type='expand'``. The resulting column names     will be the Series index.          >>> df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)        foo  bar     0    1    2     1    1    2     2    1    2          Passing ``result_type='broadcast'`` will ensure the same shape     result,
2: add_prefix(prefix: 'str')     Prefix labels with string `prefix`.          For Series, the row labels are prefixed.     For DataFrame, the column labels are prefixed.          Parameters     ----------     prefix : str         The string to add before each label.          Returns     -------     Series or DataFrame         New Series or DataFrame with updated labels.          See Also     --------     Series.add_suffix: Suffix row labels with string `suffix`.     DataFrame.add_suffix: Suffix column labels with string `suffix`.          Examples     --------     >>> s = pd.Series([1, 2, 3, 4])     >>> s     0    1     1    2     2    3     3    4     dtype: int64          >>> s.add_prefix('item_')     item_0    1     item_1    2     item_2    3     item_3    4     dtype: int64          >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})     >>> df        A  B     0  1  3     1  2  4     2  3  5     3  4  6          >>> df.add_prefix('col_')          col_A  col_B     0       1       3     1       2       4     2       3       5     3       4       6 

## Problem: 
Sample dataframe: df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on. The resulting dataframe should look like so: result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})  Notice that e is the natural constant. Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.  

## Unfinished Code Solution:
<code>
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## [insert]:
```
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())
```
"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
{examples_prompt}
\n
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
# [insert]:
"""

    sys_prompt = LLAMA_SYSTEM_PROMPT
    # prompt_template = ensemble_prompt(sys_prompt, user_prompt, model, examples=[example1, example2, example3], answers=[answer1, answer2, answer3])
    if 'gpt' in model: prompt = ['', user_prompt]
    else: prompt = user_prompt
    return prompt


# def llama_3shots_prompt_type2(ret_docs, question, model):
#     example1 = """## Potential documents:
# 0: yticks(ticks=None, labels=None, **kwargs)     Get or set the current tick locations and labels of the y-axis.          Pass no arguments to return the current values without modifying them.          Parameters     ----------     ticks : array-like, optional         The list of ytick locations.  Passing an empty list removes all yticks.     labels : array-like, optional         The labels to place at the given *ticks* locations.  This argument can         only be passed if *ticks* is passed as well.     **kwargs         `.Text` properties can be used to control the appearance of the labels.          Returns     -------     locs         The list of ytick locations.     labels         The list of ylabel `.Text` objects.          Notes     -----     Calling this function with no arguments (e.g. ``yticks()``) is the pyplot     equivalent of calling `~.Axes.get_yticks` and `~.Axes.get_yticklabels` on     the current axes.     Calling this function with arguments is the pyplot equivalent of calling     `~.Axes.set_yticks` and `~.Axes.set_yticklabels` on the current axes.          Examples     --------     >>> locs, labels = yticks()  # Get the current locations and labels.     >>> yticks(np.arange(0, 1, step=0.2))  # Set label locations.     >>> yticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.     >>> yticks([0, 1, 2], ['January', 'February', 'March'],     ...        rotation=45)  # Set text labels and properties.     >>> yticks([])  # Disable yticks.
#
#
#
# ## Unfinished Code Snippet:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# x = np.arange(2010, 2020)
# y = np.arange(10)
# plt.plot(x, y)
#
# # Set the transparency of xtick labels to be 0.5
# # SOLUTION START
# """
#     answer1 = """
# <code>
# plt.yticks(alpha=0.5)
# </code>
# """
#     example2 = """## Potential documents:
# 0: plot(*args, scalex=True, scaley=True, data=None, **kwargs)     Plot y versus x as lines and/or markers.          Call signatures::              plot([x], y, [fmt], *, data=None, **kwargs)         plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)          The coordinates of the points or line nodes are given by *x*, *y*.          The optional parameter *fmt* is a convenient way for defining basic     formatting like color, marker and linestyle. It's a shortcut string     notation described in the *Notes* section below.          >>> plot(x, y)        # plot x and y using default line style and color     >>> plot(x, y, 'bo')  # plot x and y using blue circle markers     >>> plot(y)           # plot y using x as index array 0..N-1     >>> plot(y, 'r+')     # ditto, but with red plusses          You can use `.Line2D` properties as keyword arguments for more     control on the appearance. Line properties and *fmt* can be mixed.     The following two calls yield identical results:          >>> plot(x, y, 'go--', linewidth=2, markersize=12)     >>> plot(x, y, color='green', marker='o', linestyle='dashed',     ...      linewidth=2, markersize=12)          When conflicting with *fmt*, keyword arguments take precedence.               **Plotting labelled data**          There's a convenient way for plotting objects with labelled data (i.e.     data that can be accessed by index ``obj['y']``). Instead of giving     the data in *x* and *y*, you can provide the object in the *data*     parameter and just give the labels for *x* and *y*::          >>> plot('xlabel', 'ylabel', data=obj)          All indexable objects are supported. This could e.g. be a `dict`, a     `pandas.DataFrame` or a structured numpy array.               **Plotting multiple sets of data**          There are various ways to plot multiple sets of data.          - The most straight forward way is just to call `plot` multiple times.       Example:            >>> plot(x1, y1, 'bo')       >>> plot(x2, y2, 'go')          - If *x* and/or *y* are 2D arrays a separate data set will be drawn       for every column. If both *x* and *y* are 2D, they must have the       same shape. If only one of them is 2D with shape (N, m) the other       must have length N and will be used for every data set m.            Example:            >>> x = [1, 2, 3]       >>> y = np.array([[1, 2], [3, 4], [5, 6]])       >>> plot(x, y)            is equivalent to:            >>> for col in range(y.shape[1]):       ...     plot(x, y[:, col])          - The third way is to specify multiple sets of *[x]*, *y*, *[fmt]*       groups::            >>> plot(x1, y1, 'g^', x2, y2, 'g-')            In this case, any additional keyword argument applies to all       datasets. Also this syntax cannot be combined with the *data*       parameter.          By default, each line is assigned a different style specified by a     'style cycle'. The *fmt* and line property parameters are only     necessary if you want explicit deviations from these defaults.     Alternatively, you can also change the style cycle using     :rc:`axes.prop_cycle`.               Parameters     ----------     x, y : array-like or scalar         The horizontal / vertical coordinates of the data points.         *x* values are optional and default to ``range(len(y))``.              Commonly, these parameters are 1D arrays.              They can also be scalars, or two-dimensional (in that case, the         columns represent separate data sets).              These arguments cannot be passed as keywords.          fmt : str, optional         A format string, e.g. 'ro' for red circles. See the *Notes*         section for a full description of the format strings.              Format strings are just an abbreviation for quickly setting         basic line properties. All of these and more can also be         controlled by keyword arguments.              This argument cannot be passed as keyword.          data : indexable object, optional         An object with labelled data. If given, provide the label names to         plot in *x* and *y*.              .. note::             Technically there's a slight ambiguity in calls where the             second label is a valid *fmt*. ``plot('n', 'o', data=obj)``             could be ``plt(x, y)`` or ``plt(y, fmt)``. In such cases,             the former interpretation is chosen, but a warning is issued.             You may suppress the warning by adding an empty format string             ``plot('n', 'o', '', data=obj)``.          Returns     -------     list of `.Line2D`         A list of lines representing the plotted data.          Other Parameters     ----------------     scalex, scaley : bool, default: True         These parameters determine if the view limits are adapted to the         data limits. The values are passed on to `autoscale_view`.          **kwargs : `.Line2D` properties, optional         *kwargs* are used to specify properties like a line label (for         auto legends), linewidth, antialiasing, marker face color.         Example::              >>> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)         >>> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')              If you specify multiple lines with one plot call, the kwargs apply         to all those lines. In case the label object is iterable, each         element is used as labels for each set of data.              Here is a list of available `.Line2D` properties:              Properties:         agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array         alpha: scalar or None         animated: bool         antialiased or aa: bool         clip_box: `.Bbox`         clip_on: bool         clip_path: Patch or (Path, Transform) or None         color or c: color         dash_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}         dash_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}         dashes: sequence of floats (on/off ink in points) or (None, None)         data: (2, N) array or two 1D arrays         drawstyle or ds: {'default', 'steps', 'steps-pre', 'steps-mid', 'steps-post'}, default: 'default'         figure: `.Figure`         fillstyle: {'full', 'left', 'right', 'bottom', 'top', 'none'}         gid: str         in_layout: bool         label: object         linestyle or ls: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}         linewidth or lw: float         marker: marker style string, `~.path.Path` or `~.markers.MarkerStyle`         markeredgecolor or mec: color         markeredgewidth or mew: float         markerfacecolor or mfc: color         markerfacecoloralt or mfcalt: color         markersize or ms: float         markevery: None or int or (int, int) or slice or list[int] or float or (float, float) or list[bool]         path_effects: `.AbstractPathEffect`         picker: float or callable[[Artist, Event], tuple[bool, dict]]         pickradius: float         rasterized: bool         sketch_params: (scale: float, length: float, randomness: float)         snap: bool or None         solid_capstyle: `.CapStyle` or {'butt', 'projecting', 'round'}         solid_joinstyle: `.JoinStyle` or {'miter', 'round', 'bevel'}         transform: unknown         url: str         visible: bool         xdata: 1D array         ydata: 1D array         zorder: float          See Also     --------     scatter : XY scatter plot with markers of varying size and/or color (         sometimes also called bubble chart).          Notes     -----     **Format Strings**          A format string consists of a part for color, marker and line::              fmt = '[marker][line][color]'          Each of them is optional. If not provided, the value from the style     cycle is used. Exception: If ``line`` is given, but no ``marker``,     the data will be a line without markers.          Other combinations such as ``[color][marker][line]`` are also     supported, but note that their parsing may be ambiguous.          **Markers**          =============   ===============================     character       description     =============   ===============================     ``'.'``         point marker     ``','``         pixel marker     ``'o'``         circle marker     ``'v'``         triangle_down marker     ``'^'``         triangle_up marker     ``'<'``         triangle_left marker     ``'>'``         triangle_right marker     ``'1'``         tri_down marker     ``'2'``         tri_up marker     ``'3'``         tri_left marker     ``'4'``         tri_right marker     ``'8'``         octagon marker     ``'s'``         square marker     ``'p'``         pentagon marker     ``'P'``         plus (filled) marker     ``'*'``         star marker     ``'h'``         hexagon1 marker     ``'H'``         hexagon2 marker     ``'+'``         plus marker     ``'x'``         x marker     ``'X'``         x (filled) marker     ``'D'``         diamond marker     ``'d'``         thin_diamond marker     ``'|'``         vline marker     ``'_'``         hline marker     =============   ===============================          **Line Styles**          =============    ===============================     character        description     =============    ===============================     ``'-'``          solid line style     ``'--'``         dashed line style     ``'-.'``         dash-dot line style     ``':'``          dotted line style     =============    ===============================          Example format strings::              'b'    # blue markers with default shape         'or'   # red circles         '-g'   # green solid line         '--'   # dashed line with default color         '^k:'  # black triangle_up markers connected by a dotted line          **Colors**          The supported color abbreviations are the single letter codes          =============    ===============================     character        color     =============    ===============================     ``'b'``          blue     ``'g'``          green     ``'r'``          red     ``'c'``          cyan     ``'m'``          magenta     ``'y'``          yellow     ``'k'``          black     ``'w'``          white     =============    ===============================          and the ``'CN'`` colors that index into the default property cycle.          If the color is the only part of the format string, you can     additionally use any  `matplotlib.colors` spec, e.g. full names     (``'green'``) or hex strings (``'#008000'``).
# 1: tick_params(axis='both', **kwargs)     Change the appearance of ticks, tick labels, and gridlines.          Tick properties that are not explicitly set using the keyword     arguments remain unchanged unless *reset* is True.          Parameters     ----------     axis : {'x', 'y', 'both'}, default: 'both'         The axis to which the parameters are applied.     which : {'major', 'minor', 'both'}, default: 'major'         The group of ticks to which the parameters are applied.     reset : bool, default: False         Whether to reset the ticks to defaults before updating them.          Other Parameters     ----------------     direction : {'in', 'out', 'inout'}         Puts ticks inside the axes, outside the axes, or both.     length : float         Tick length in points.     width : float         Tick width in points.     color : color         Tick color.     pad : float         Distance in points between tick and label.     labelsize : float or str         Tick label font size in points or as a string (e.g., 'large').     labelcolor : color         Tick label color.     colors : color         Tick color and label color.     zorder : float         Tick and label zorder.     bottom, top, left, right : bool         Whether to draw the respective ticks.     labelbottom, labeltop, labelleft, labelright : bool         Whether to draw the respective tick labels.     labelrotation : float         Tick label rotation     grid_color : color         Gridline color.     grid_alpha : float         Transparency of gridlines: 0 (transparent) to 1 (opaque).     grid_linewidth : float         Width of gridlines in points.     grid_linestyle : str         Any valid `.Line2D` line style spec.          Examples     --------     ::              ax.tick_params(direction='out', length=6, width=2, colors='r',                        grid_color='r', grid_alpha=0.5)          This will make all major ticks be red, pointing out of the box,     and with dimensions 6 points by 2 points.  Tick labels will     also be red.  Gridlines will be red and translucent.
#
#
#
# ## Unfinished Code Snippet:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# x = np.arange(10)
# y = np.arange(10)
#
# # Plot y over x in a line chart. Show x axis tick labels but hide the x axis ticks
# # SOLUTION START
# """
#     answer2 = """
# <code>
# plt.plot(x, y)
# plt.tick_params(bottom=False, labelbottom=True)
# </code>
# """
#     example3 = """## Potential documents:
# 0: catplot(*, x=None, y=None, hue=None, data=None, row=None, col=None, col_wrap=None, estimator=<function mean at 0x7f365754e3b0>, ci=95, n_boot=1000, units=None, seed=None, order=None, hue_order=None, row_order=None, col_order=None, kind='strip', height=5, aspect=1, orient=None, color=None, palette=None, legend=True, legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)     Figure-level interface for drawing categorical plots onto a FacetGrid.          This function provides access to several axes-level functions that     show the relationship between a numerical and one or more categorical     variables using one of several visual representations. The ``kind``     parameter selects the underlying axes-level function to use:          Categorical scatterplots:          - :func:`stripplot` (with ``kind="strip"``; the default)     - :func:`swarmplot` (with ``kind="swarm"``)          Categorical distribution plots:          - :func:`boxplot` (with ``kind="box"``)     - :func:`violinplot` (with ``kind="violin"``)     - :func:`boxenplot` (with ``kind="boxen"``)          Categorical estimate plots:          - :func:`pointplot` (with ``kind="point"``)     - :func:`barplot` (with ``kind="bar"``)     - :func:`countplot` (with ``kind="count"``)          Extra keyword arguments are passed to the underlying function, so you     should refer to the documentation for each to see kind-specific options.          Note that unlike when using the axes-level functions directly, data must be     passed in a long-form DataFrame with variables specified by passing strings     to ``x``, ``y``, ``hue``, etc.          As in the case with the underlying plot functions, if variables have a     ``categorical`` data type, the levels of the categorical variables, and     their order will be inferred from the objects. Otherwise you may have to     use alter the dataframe sorting or use the function parameters (``orient``,     ``order``, ``hue_order``, etc.) to set up the plot correctly.          This function always treats one of the variables as categorical and     draws data at ordinal positions (0, 1, ... n) on the relevant axis, even     when the data has a numeric or date type.          See the :ref:`tutorial <categorical_tutorial>` for more information.              After plotting, the :class:`FacetGrid` with the plot is returned and can     be used directly to tweak supporting plot details or add other layers.          Parameters     ----------     x, y, hue : names of variables in ``data``         Inputs for plotting long-form data. See examples for interpretation.             data : DataFrame         Long-form (tidy) dataset for plotting. Each column should correspond         to a variable, and each row should correspond to an observation.         row, col : names of variables in ``data``, optional         Categorical variables that will determine the faceting of the grid.     col_wrap : int         "Wrap" the column variable at this width, so that the column facets         span multiple rows. Incompatible with a ``row`` facet.         estimator : callable that maps vector -> scalar, optional         Statistical function to estimate within each categorical bin.     ci : float or "sd" or None, optional         Size of confidence intervals to draw around estimated values.  If         "sd", skip bootstrapping and draw the standard deviation of the         observations. If ``None``, no bootstrapping will be performed, and         error bars will not be drawn.     n_boot : int, optional         Number of bootstrap iterations to use when computing confidence         intervals.     units : name of variable in ``data`` or vector data, optional         Identifier of sampling units, which will be used to perform a         multilevel bootstrap and account for repeated measures design.     seed : int, numpy.random.Generator, or numpy.random.RandomState, optional         Seed or random number generator for reproducible bootstrapping.         order, hue_order : lists of strings, optional         Order to plot the categorical levels in, otherwise the levels are         inferred from the data objects.             row_order, col_order : lists of strings, optional         Order to organize the rows and/or columns of the grid in, otherwise the         orders are inferred from the data objects.     kind : str, optional         The kind of plot to draw, corresponds to the name of a categorical         axes-level plotting function. Options are: "strip", "swarm", "box", "violin",         "boxen", "point", "bar", or "count".     height : scalar         Height (in inches) of each facet. See also: ``aspect``.         aspect : scalar         Aspect ratio of each facet, so that ``aspect * height`` gives the width         of each facet in inches.         orient : "v" | "h", optional         Orientation of the plot (vertical or horizontal). This is usually         inferred based on the type of the input variables, but it can be used         to resolve ambiguity when both `x` and `y` are numeric or when         plotting wide-form data.         color : matplotlib color, optional         Color for all of the elements, or seed for a gradient palette.         palette : palette name, list, or dict         Colors to use for the different levels of the ``hue`` variable. Should         be something that can be interpreted by :func:`color_palette`, or a         dictionary mapping hue levels to matplotlib colors.         legend : bool, optional         If ``True`` and there is a ``hue`` variable, draw a legend on the plot.     legend_out : bool         If ``True``, the figure size will be extended, and the legend will be         drawn outside the plot on the center right.         share{x,y} : bool, 'col', or 'row' optional         If true, the facets will share y axes across columns and/or x axes         across rows.         margin_titles : bool         If ``True``, the titles for the row variable are drawn to the right of         the last column. This option is experimental and may not work in all         cases.         facet_kws : dict, optional         Dictionary of other keyword arguments to pass to :class:`FacetGrid`.     kwargs : key, value pairings         Other keyword arguments are passed through to the underlying plotting         function.          Returns     -------     g : :class:`FacetGrid`         Returns the :class:`FacetGrid` object with the plot on it for further         tweaking.          Examples     --------          Draw a single facet to use the :class:`FacetGrid` legend placement:          .. plot::         :context: close-figs              >>> import seaborn as sns         >>> sns.set_theme(style="ticks")         >>> exercise = sns.load_dataset("exercise")         >>> g = sns.catplot(x="time", y="pulse", hue="kind", data=exercise)          Use a different plot kind to visualize the same data:          .. plot::         :context: close-figs              >>> g = sns.catplot(x="time", y="pulse", hue="kind",         ...                data=exercise, kind="violin")          Facet along the columns to show a third categorical variable:          .. plot::         :context: close-figs              >>> g = sns.catplot(x="time", y="pulse", hue="kind",         ...                 col="diet", data=exercise)          Use a different height and aspect ratio for the facets:          .. plot::         :context: close-figs              >>> g = sns.catplot(x="time", y="pulse", hue="kind",         ...                 col="diet", data=exercise,         ...                 height=5, aspect=.8)          Make many column facets and wrap them into the rows of the grid:          .. plot::         :context: close-figs              >>> titanic = sns.load_dataset("titanic")         >>> g = sns.catplot(x="alive", col="deck", col_wrap=4,         ...                 data=titanic[titanic.deck.notnull()],         ...                 kind="count", height=2.5, aspect=.8)          Plot horizontally and pass other keyword arguments to the plot function:          .. plot::         :context: close-figs              >>> g = sns.catplot(x="age", y="embark_town",         ...                 hue="sex", row="class",         ...                 data=titanic[titanic.embark_town.notnull()],         ...                 orient="h", height=2, aspect=3, palette="Set3",         ...                 kind="violin", dodge=True, cut=0, bw=.2)          Use methods on the returned :class:`FacetGrid` to tweak the presentation:          .. plot::         :context: close-figs              >>> g = sns.catplot(x="who", y="survived", col="class",         ...                 data=titanic, saturation=.5,         ...                 kind="bar", ci=None, aspect=.6)         >>> (g.set_axis_labels("", "Survival Rate")         ...   .set_xticklabels(["Men", "Women", "Children"])         ...   .set_titles("{col_name} {col_var}")         ...   .set(ylim=(0, 1))         ...   .despine(left=True))  #doctest: +ELLIPSIS         <seaborn.axisgrid.FacetGrid object at 0x...>
# 1: flatten()     a.flatten(order='C')          Return a copy of the array collapsed into one dimension.          Parameters     ----------     order : {'C', 'F', 'A', 'K'}, optional         'C' means to flatten in row-major (C-style) order.         'F' means to flatten in column-major (Fortran-         style) order. 'A' means to flatten in column-major         order if `a` is Fortran *contiguous* in memory,         row-major order otherwise. 'K' means to flatten         `a` in the order the elements occur in memory.         The default is 'C'.          Returns     -------     y : ndarray         A copy of the input array, flattened to one dimension.          See Also     --------     ravel : Return a flattened array.     flat : A 1-D flat iterator over the array.          Examples     --------     >>> a = np.array([[1,2], [3,4]])     >>> a.flatten()     array([1, 2, 3, 4])     >>> a.flatten('F')     array([1, 3, 2, 4])
# 2: set_ylabel(ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)     Set the label for the y-axis.          Parameters     ----------     ylabel : str         The label text.          labelpad : float, default: :rc:`axes.labelpad`         Spacing in points from the Axes bounding box including ticks         and tick labels.  If None, the previous value is left as is.          loc : {'bottom', 'center', 'top'}, default: :rc:`yaxis.labellocation`         The label position. This is a high-level alternative for passing         parameters *y* and *horizontalalignment*.          Other Parameters     ----------------     **kwargs : `.Text` properties         `.Text` properties control the appearance of the label.          See Also     --------     text : Documents the properties supported by `.Text`.
#
#
#
# ## Unfinished Code Snippet:
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# df = sns.load_dataset("exercise")
#
# # Make catplots of scatter plots by using "time" as x, "pulse" as y, "kind" as hue, and "diet" as col
# # Do not show any ylabel on either subplot
# # SOLUTION START
# """
#     answer3 = """
# <code>
# g = sns.catplot(x="time", y="pulse", hue="kind", col="diet", data=df)
# axs = g.axes.flatten()
# axs[0].set_ylabel("")
# </code>
# """
#     potential_docs, prompt, answer = process_docs_question(ret_docs, question)
#     user_prompt = f"""
# ## Potential documents:
# {potential_docs}
# \n
# ## Unfinished Code Snippet:
# {prompt}
# """
#
#     sys_prompt = LLAMA_SYSTEM_PROMPT_TYPE2
#     if model.startswith('llama2') or model.startswith('codellama'):
#         prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n {example1} [/INST] {answer1}</s>\n
# <s>[INST] {example2} [/INST] {answer2}</s>\n
# <s>[INST] {example3} [/INST] {answer3}</s>\n
# <s>[INST] {user_prompt} [/INST]
# """
#     elif model.startswith('llama3'):
#         prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{example1}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>{answer1}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{example2}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>{answer2}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{example3}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>{answer3}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>
# """
#
#     return prompt_template


# def prompt_0shot(ret_docs, question, model):
#     if '\nA:' in question:
#         return llama_0shot_prompt_type1(ret_docs, question, model)
#     else:
#         return llama_0shot_prompt_type2(ret_docs, question, model)



examples_least_to_most = """
## Potential documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)     Returns a boolean array where two arrays are element-wise equal within a     tolerance.          The tolerance values are positive, typically very small numbers.  The     relative difference (`rtol` * abs(`b`)) and the absolute difference     `atol` are added together to compare against the absolute difference     between `a` and `b`.          .. warning:: The default `atol` is not appropriate for comparing numbers                  that are much smaller than one (see Notes).          Parameters     ----------     a, b : array_like         Input arrays to compare.     rtol : float         The relative tolerance parameter (see Notes).     atol : float         The absolute tolerance parameter (see Notes).     equal_nan : bool         Whether to compare NaN's as equal.  If True, NaN's in `a` will be         considered equal to NaN's in `b` in the output array.          Returns     -------     y : array_like         Returns a boolean array of where `a` and `b` are equal within the         given tolerance. If both `a` and `b` are scalars, returns a single         boolean value.          See Also     --------     allclose     math.isclose          Notes     -----     .. versionadded:: 1.7.0          For finite values, isclose uses the following equation to test whether     two floating point values are equivalent.           absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))          Unlike the built-in `math.isclose`, the above equation is not symmetric     in `a` and `b` -- it assumes `b` is the reference value -- so that     `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,     the default value of atol is not zero, and is used to determine what     small values should be considered close to zero. The default value is     appropriate for expected values of order unity: if the expected values     are significantly smaller than one, it can result in false positives.     `atol` should be carefully selected for the use case at hand. A zero value     for `atol` will result in `False` if either `a` or `b` is zero.          `isclose` is not defined for non-numeric data types.          Examples     --------     >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])     array([ True, False])     >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])     array([ True, True])     >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])     array([False,  True])     >>> np.isclose([1.0, np.nan], [1.0, np.nan])     array([ True, False])     >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)     array([ True, True])     >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])     array([ True, False])
1: sum(iterable, start=0)     Return the sum of a 'start' value (default: 0) plus an iterable of numbers          When the iterable is empty, return the start value.     This function is intended specifically for use with numeric values and may     reject non-numeric types.  

## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed). Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values. Here is a simple standalone example to illustrate this issue : import numpy as np n = 10 m = 4 tag = np.random.rand(n, m) s1 = np.sum(tag, axis=1) s2 = np.sum(tag[:, ::-1], axis=1) # print the number of times s1 is not equal to s2 (should be 0) print np.nonzero(s1 != s2)[0].shape[0] If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance. The problem is I need to use those in functions like np.in1d where I can't really give a tolerance... What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above. Is there a way to avoid this issue? 

## Unfinished Code Solution:
<code>
import numpy as np
n = 20
m = 10
tag = np.random.rand(n, m)
s1 = np.sum(tag, axis=1)
s2 = np.sum(tag[:, ::-1], axis=1)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
Step 1: Decompose the problem into subquestions:
1. How to compare two arrays element-wise within a tolerance?
2. How to count the number of elements that are not equal within the tolerance?

Step 2: Answer each subquestion:
1. How to compare two arrays element-wise within a tolerance?
    - `numpy.isclose()` can be used to compare two arrays element-wise within a tolerance.

2. How to count the number of elements that are not equal within the tolerance?
    - Use `numpy.sum()` on the boolean array returned by `numpy.isclose()` to count the number of elements that are equal within the tolerance, and subtract this count from the total number of elements to get the number of truly different elements.

Step 3: Write the final program:
To compare `s1` and `s2` within a tolerance and count the number of truly different elements
the code in [insert] should be:
```
comparison = np.isclose(s1,s2)
result = np.sum(~comparison)
```



## Potential documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)     Gather slices from `params` into a Tensor with shape specified by `indices`.          `indices` is a `Tensor` of indices into `params`. The index vectors are     arranged along the last axis of `indices`.          This is similar to `tf.gather`, in which `indices` defines slices into the     first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the     first `N` dimensions of `params`, where `N = indices.shape[-1]`.          Caution: On CPU, if an out of bound index is found, an error is returned.     On GPU, if an out of bound index is found, a 0 is stored in the     corresponding output value.          ## Gathering scalars          In the simplest case the vectors in `indices` index the full rank of `params`:          >>> tf.gather_nd(     ...     indices=[[0, 0],     ...              [1, 1]],     ...     params = [['a', 'b'],     ...               ['c', 'd']]).numpy()     array([b'a', b'd'], dtype=object)          In this case the result has 1-axis fewer than `indices`, and each index vector     is replaced by the scalar indexed from `params`.          In this case the shape relationship is:          ```     index_depth = indices.shape[-1]     assert index_depth == params.shape.rank     result_shape = indices.shape[:-1]     ```          If `indices` has a rank of `K`, it is helpful to think `indices` as a     (K-1)-dimensional tensor of indices into `params`.          ## Gathering slices          If the index vectors do not index the full rank of `params` then each location     in the result contains a slice of params. This example collects rows from a     matrix:          >>> tf.gather_nd(     ...     indices = [[1],     ...                [0]],     ...     params = [['a', 'b', 'c'],     ...               ['d', 'e', 'f']]).numpy()     array([[b'd', b'e', b'f'],            [b'a', b'b', b'c']], dtype=object)          Here `indices` contains `[2]` index vectors, each with a length of `1`.     The index vectors each refer to rows of the `params` matrix. Each     row has a shape of `[3]` so the output shape is `[2, 3]`.          In this case, the relationship between the shapes is:          ```     index_depth = indices.shape[-1]     outer_shape = indices.shape[:-1]     assert index_depth <= params.shape.rank     inner_shape = params.shape[index_depth:]     output_shape = outer_shape + inner_shape     ```          It is helpful to think of the results in this case as tensors-of-tensors.     The shape of the outer tensor is set by the leading dimensions of `indices`.     While the shape of the inner tensors is the shape of a single slice.          ## Batches          Additionally both `params` and `indices` can have `M` leading batch     dimensions that exactly match. In this case `batch_dims` must be set to `M`.          For example, to collect one row from each of a batch of matrices you could     set the leading elements of the index vectors to be their location in the     batch:

## Problem: 
I'm using tensorflow 2.10.0.  import tensorflow as tf x = [[1,2,3],[4,5,6]] y = [0,1] z = [1,2] x = tf.constant(x) y = tf.constant(y) z = tf.constant(z) m = x[y,z]  What I expect is m = [2,6] I can get the result by theano or numpy. How I get the result using tensorflow?   

## Unfinished Code Solution:
<code>
import tensorflow as tf


x = [[1,2,3],[4,5,6]]
y = [0,1]
z = [1,2]
x = tf.constant(x)
y = tf.constant(y)
z = tf.constant(z)
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
Step 1: Decompose the problem into subquestions:
1. How to use `tf.gather_nd` to index elements from `params` using `indices`.
2. How to construct the appropriate `indices` tensor to retrieve the specified elements from `params`.

Step 2: Answer each subquestion:
1. How to use `tf.gather_nd` to index elements from `params` using `indices`:
   - `tf.gather_nd` can be used to gather specific elements from a tensor according to the indices provided.

2. How to construct the appropriate `indices` tensor to retrieve the specified elements from `params`:
   - `indices` should be a tensor that specifies the index positions in `params` to be gathered. Each row in `indices` tensor should specify the position of an element in `params`.

Step 3: Write the final program:
To gather elements `[2, 6]` from `x` using indices `y` and `z`, the code in [insert] should construct the `indices` tensor by combining `y` and `z` and use `tf.gather_nd` to gather the required elements.
the code in [insert] should be:
```
indices = tf.stack([y, z], axis=1)
result = tf.gather_nd(x, indices)
```



## Potential documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)     Join columns of another DataFrame.          Join columns with `other` DataFrame either on index or on a key     column. Efficiently join multiple DataFrame objects by index at once by     passing a list.          Parameters     ----------     other : DataFrame, Series, or list of DataFrame         Index should be similar to one of the columns in this one. If a         Series is passed, its name attribute must be set, and that will be         used as the column name in the resulting joined DataFrame.     on : str, list of str, or array-like, optional         Column or index level name(s) in the caller to join on the index         in `other`, otherwise joins index-on-index. If multiple         values given, the `other` DataFrame must have a MultiIndex. Can         pass an array as the join key if it is not already contained in         the calling DataFrame. Like an Excel VLOOKUP operation.     how : {'left', 'right', 'outer', 'inner'}, default 'left'         How to handle the operation of the two objects.              * left: use calling frame's index (or column if on is specified)         * right: use `other`'s index.         * outer: form union of calling frame's index (or column if on is           specified) with `other`'s index, and sort it.           lexicographically.         * inner: form intersection of calling frame's index (or column if           on is specified) with `other`'s index, preserving the order           of the calling's one.     lsuffix : str, default ''         Suffix to use from left frame's overlapping columns.     rsuffix : str, default ''         Suffix to use from right frame's overlapping columns.     sort : bool, default False         Order result DataFrame lexicographically by the join key. If False,         the order of the join key depends on the join type (how keyword).          Returns     -------     DataFrame         A dataframe containing columns from both the caller and `other`.          See Also     --------     DataFrame.merge : For column(s)-on-column(s) operations.          Notes     -----     Parameters `on`, `lsuffix`, and `rsuffix` are not supported when     passing a list of `DataFrame` objects.          Support for specifying index levels as the `on` parameter was added     in version 0.23.0.          Examples     --------     >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],     ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})          >>> df       key   A     0  K0  A0     1  K1  A1     2  K2  A2     3  K3  A3     4  K4  A4     5  K5  A5          >>> other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],     ...                       'B': ['B0', 'B1', 'B2']})          >>> other       key   B     0  K0  B0     1  K1  B1     2  K2  B2          Join DataFrames using their indexes.
1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis
2: add_prefix(prefix: 'str')     Prefix labels with string `prefix`.          For Series, the row labels are prefixed.     For DataFrame, the column labels are prefixed.          Parameters     ----------     prefix : str         The string to add before each label.          Returns     -------     Series or DataFrame         New Series or DataFrame with updated labels.          See Also     --------     Series.add_suffix: Suffix row labels with string `suffix`.     DataFrame.add_suffix: Suffix column labels with string `suffix`.          Examples     --------     >>> s = pd.Series([1, 2, 3, 4])     >>> s     0    1     1    2     2    3     3    4     dtype: int64          >>> s.add_prefix('item_')     item_0    1     item_1    2     item_2    3     item_3    4     dtype: int64          >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})     >>> df        A  B     0  1  3     1  2  4     2  3  5     3  4  6          >>> df.add_prefix('col_')          col_A  col_B     0       1       3     1       2       4     2       3       5     3       4       6 

## Problem: 
Sample dataframe: df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on. The resulting dataframe should look like so: result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})  Notice that e is the natural constant. Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.  

## Unfinished Code Solution:
<code>
import pandas as pd


df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
</code>
BEGIN SOLUTION
<code>
[insert]
</code>
END SOLUTION
<code>
print(result)
</code>

## Code Generation for [insert]:
Step 1: Decompose the problem into sub-questions
1. How to compute the sigmoid of each value in the columns?
2. How to add new columns to the DataFrame with the corresponding sigmoid values?
3. How to correctly name the new columns with a prefix?
4. How to combine the original dataframe with the new sigmoid columns?

Step 2: Answer each sub-question
1. How to compute the sigmoid of each value in the columns?
    - We can use the sigmoid function, which can be defined as `1 / (1 + exp(-x))`.

2. How to add new columns to the DataFrame with the corresponding sigmoid values?
    - We can use DataFrame's apply method for this purpose, which allows applying a function to each column.

3. How to correctly name the new columns with a prefix?
    - We can use DataFrame's add_prefix method to create new column names with the desired prefix.
    
4. How to combine the original dataframe with the new sigmoid columns?
    - We can use the DataFrame's join method to achieve this
    
Step 3: Write the final program
The program should first define a sigmoid function, and use df.apply, then add prefix using df.add_prefix, and finally use df.join to combine the new columns
the code in [insert] should be:
```
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid_df = df.apply(sigmoid)
sigmoid_df = sigmoid_df.add_prefix('sigmoid_')
result = df.join(sigmoid_df)
```
"""

def prompt_least_to_most(ret_docs, question, model):
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)

    user_prompt = f"""
{examples_least_to_most}
\n
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}

## Code Generation for [insert]:
"""
    # sys_prompt = 'You should only generate the code in [insert]'
    # prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    prompt = [SYS_PROMPT_LEAST_TO_MOST, user_prompt] if 'gpt' in model else user_prompt
    return prompt


def prompt_plan_and_solve(ret_docs, question, model):
    plan_and_solve_prompt = """You are a senior Python programmer, given some Potential Documents, a Problem and its Unfinished Code Solution,
Your task is to first understand the problem and devise a plan to complete the code Solution.
Second, you should carry out the plan, and complete the Unfinished Code Solution tagged with ```
"""
    potential_docs, prompt, answer = process_docs_question(ret_docs, question)

    user_prompt = f"""
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
"""
    sys_prompt = plan_and_solve_prompt
    prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt


def prompt_RaR(ret_docs, question, model):
    RaR_prompt = """You are a senior python programmer, given some potential documents, a problem and its unfinished code solution
Your task is to first rephrase and expand the Problem, then complete the Unfinished Code Solution without modifying its existing part"""

    potential_docs, prompt, answer = process_docs_question(ret_docs, question)
    user_prompt = f"""
## Potential documents:
{potential_docs}
## Problem: 
{prompt}

## Unfinished Code Solution:
{answer}
"""
    sys_prompt = RaR_prompt
    prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
    return prompt



if __name__ == '__main__':
    """
    get examples
    """
    import sys, platform
    import random

    system = platform.system()
    if system == 'Darwin':
        root_path = '/Users/zhaoshengming/Code_RAG_Benchmark'
    elif system == 'Linux':
        root_path = '/home/zhaoshengming/Code_RAG_Benchmark'
    sys.path.insert(0, root_path)
    from dataset_utils.DS1000_utils import DS1000Loader
    from generator.generate_utils import truncate_docs
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


    # print(llama_3shots_prompt(ret_docs=['asda', 'asdawr'], question=ds1000['Matplotlib'][0]['prompt'], model='codellama-13b-instruct'))
