def llama_3shot_prompt(ret_docs, question, model):
    assert model in ['llama2-13b-chat', 'codellama-13b-instruct', 'llama3-8b']
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
    prompt, answer = question.split('\nA:')
    prompt = prompt.replace('Problem:', '').replace('\n', ' ')

    sys_prompt = """You are a senior python programmer, given some potential api documents starts with `## Potential documents`, a program description starts with `## Problem`, and the unfinished code solution starts with `## Unfinished Code Solution`, 
you should first read the potential documents, and then use the knowledge in documents to complete the code solution according to the problem.
you should only output the completed code solution
\n
"""

    example1 = """## Potential documents:
0: isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)     Returns a boolean array where two arrays are element-wise equal within a     tolerance.          The tolerance values are positive, typically very small numbers.  The     relative difference (`rtol` * abs(`b`)) and the absolute difference     `atol` are added together to compare against the absolute difference     between `a` and `b`.          .. warning:: The default `atol` is not appropriate for comparing numbers                  that are much smaller than one (see Notes).          Parameters     ----------     a, b : array_like         Input arrays to compare.     rtol : float         The relative tolerance parameter (see Notes).     atol : float         The absolute tolerance parameter (see Notes).     equal_nan : bool         Whether to compare NaN's as equal.  If True, NaN's in `a` will be         considered equal to NaN's in `b` in the output array.          Returns     -------     y : array_like         Returns a boolean array of where `a` and `b` are equal within the         given tolerance. If both `a` and `b` are scalars, returns a single         boolean value.          See Also     --------     allclose     math.isclose          Notes     -----     .. versionadded:: 1.7.0          For finite values, isclose uses the following equation to test whether     two floating point values are equivalent.           absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))          Unlike the built-in `math.isclose`, the above equation is not symmetric     in `a` and `b` -- it assumes `b` is the reference value -- so that     `isclose(a, b)` might be different from `isclose(b, a)`. Furthermore,     the default value of atol is not zero, and is used to determine what     small values should be considered close to zero. The default value is     appropriate for expected values of order unity: if the expected values     are significantly smaller than one, it can result in false positives.     `atol` should be carefully selected for the use case at hand. A zero value     for `atol` will result in `False` if either `a` or `b` is zero.          `isclose` is not defined for non-numeric data types.          Examples     --------     >>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])     array([ True, False])     >>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])     array([ True, True])     >>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])     array([False,  True])     >>> np.isclose([1.0, np.nan], [1.0, np.nan])     array([ True, False])     >>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)     array([ True, True])     >>> np.isclose([1e-8, 1e-7], [0.0, 0.0])     array([ True, False])     >>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)     array([False, False])     >>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])     array([ True,  True])     >>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)     array([False,  True])  
1: sum(iterable, start=0)     Return the sum of a 'start' value (default: 0) plus an iterable of numbers          When the iterable is empty, return the start value.     This function is intended specifically for use with numeric values and may     reject non-numeric types.  
\n
## Problem: 
I have an array of random floats and I need to compare it to another one that has the same values in a different order. For that matter I use the sum, product (and other combinations depending on the dimension of the table hence the number of equations needed). Nevertheless, I encountered a precision issue when I perform the sum (or product) on the array depending on the order of the values. Here is a simple standalone example to illustrate this issue : import numpy as np n = 10 m = 4 tag = np.random.rand(n, m) s1 = np.sum(tag, axis=1) s2 = np.sum(tag[:, ::-1], axis=1) # print the number of times s1 is not equal to s2 (should be 0) print np.nonzero(s1 != s2)[0].shape[0] If you execute this code it sometimes tells you that s1 and s2 are not equal and the differents is of magnitude of the computer precision. However, such elements should be considered as equal under this circumstance. The problem is I need to use those in functions like np.in1d where I can't really give a tolerance... What I want as the result is the number of truly different elements in s1 and s2, as shown in code snippet above. Is there a way to avoid this issue? 
\n
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

"""
    answer1 = """
<code>
result = (~np.isclose(s1,s2)).sum()
</code>
"""

    example2 = """## Potential documents:
0: gather_nd_v2(params, indices, batch_dims=0, name=None)     Gather slices from `params` into a Tensor with shape specified by `indices`.          `indices` is a `Tensor` of indices into `params`. The index vectors are     arranged along the last axis of `indices`.          This is similar to `tf.gather`, in which `indices` defines slices into the     first dimension of `params`. In `tf.gather_nd`, `indices` defines slices into the     first `N` dimensions of `params`, where `N = indices.shape[-1]`.          Caution: On CPU, if an out of bound index is found, an error is returned.     On GPU, if an out of bound index is found, a 0 is stored in the     corresponding output value.          ## Gathering scalars          In the simplest case the vectors in `indices` index the full rank of `params`:          >>> tf.gather_nd(     ...     indices=[[0, 0],     ...              [1, 1]],     ...     params = [['a', 'b'],     ...               ['c', 'd']]).numpy()     array([b'a', b'd'], dtype=object)          In this case the result has 1-axis fewer than `indices`, and each index vector     is replaced by the scalar indexed from `params`.          In this case the shape relationship is:          ```     index_depth = indices.shape[-1]     assert index_depth == params.shape.rank     result_shape = indices.shape[:-1]     ```          If `indices` has a rank of `K`, it is helpful to think `indices` as a     (K-1)-dimensional tensor of indices into `params`.          ## Gathering slices          If the index vectors do not index the full rank of `params` then each location     in the result contains a slice of params. This example collects rows from a     matrix:          >>> tf.gather_nd(     ...     indices = [[1],     ...                [0]],     ...     params = [['a', 'b', 'c'],     ...               ['d', 'e', 'f']]).numpy()     array([[b'd', b'e', b'f'],            [b'a', b'b', b'c']], dtype=object)          Here `indices` contains `[2]` index vectors, each with a length of `1`.     The index vectors each refer to rows of the `params` matrix. Each     row has a shape of `[3]` so the output shape is `[2, 3]`.          In this case, the relationship between the shapes is:          ```     index_depth = indices.shape[-1]     outer_shape = indices.shape[:-1]     assert index_depth <= params.shape.rank     inner_shape = params.shape[index_depth:]     output_shape = outer_shape + inner_shape     ```          It is helpful to think of the results in this case as tensors-of-tensors.     The shape of the outer tensor is set by the leading dimensions of `indices`.     While the shape of the inner tensors is the shape of a single slice.          ## Batches          Additionally both `params` and `indices` can have `M` leading batch     dimensions that exactly match. In this case `batch_dims` must be set to `M`.          For example, to collect one row from each of a batch of matrices you could     set the leading elements of the index vectors to be their location in the     batch:          >>> tf.gather_nd(     ...     indices = [[0, 1],     ...                [1, 0],     ...                [2, 4],     ...                [3, 2],     ...                [4, 1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          The `batch_dims` argument lets you omit those leading location dimensions     from the index:          >>> tf.gather_nd(     ...     batch_dims=1,     ...     indices = [[1],     ...                [0],     ...                [4],     ...                [2],     ...                [1]],     ...     params=tf.zeros([5, 7, 3])).shape.as_list()     [5, 3]          This is equivalent to caling a separate `gather_nd` for each location in the     batch dimensions.               >>> params=tf.zeros([5, 7, 3])     >>> indices=tf.zeros([5, 1])     >>> batch_dims = 1     >>>     >>> index_depth = indices.shape[-1]     >>> batch_shape = indices.shape[:batch_dims]     >>> assert params.shape[:batch_dims] == batch_shape     >>> outer_shape = indices.shape[batch_dims:-1]
\n
## Problem: 
I'm using tensorflow 2.10.0.  import tensorflow as tf x = [[1,2,3],[4,5,6]] y = [0,1] z = [1,2] x = tf.constant(x) y = tf.constant(y) z = tf.constant(z) m = x[y,z]  What I expect is m = [2,6] I can get the result by theano or numpy. How I get the result using tensorflow?   
\n
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

"""
    answer2 = """
<code>
reference code def g(x,y,z):
    return tf.gather_nd(x, [y, z])

result = g(x.__copy__(),y.__copy__(),z.__copy__())
</code>
"""
    example3 = """## Potential documents:
0: join(other: 'FrameOrSeriesUnion', on: 'IndexLabel | None' = None, how: 'str' = 'left', lsuffix: 'str' = '', rsuffix: 'str' = '', sort: 'bool' = False)     Join columns of another DataFrame.          Join columns with `other` DataFrame either on index or on a key     column. Efficiently join multiple DataFrame objects by index at once by     passing a list.          Parameters     ----------     other : DataFrame, Series, or list of DataFrame         Index should be similar to one of the columns in this one. If a         Series is passed, its name attribute must be set, and that will be         used as the column name in the resulting joined DataFrame.     on : str, list of str, or array-like, optional         Column or index level name(s) in the caller to join on the index         in `other`, otherwise joins index-on-index. If multiple         values given, the `other` DataFrame must have a MultiIndex. Can         pass an array as the join key if it is not already contained in         the calling DataFrame. Like an Excel VLOOKUP operation.     how : {'left', 'right', 'outer', 'inner'}, default 'left'         How to handle the operation of the two objects.              * left: use calling frame's index (or column if on is specified)         * right: use `other`'s index.         * outer: form union of calling frame's index (or column if on is           specified) with `other`'s index, and sort it.           lexicographically.         * inner: form intersection of calling frame's index (or column if           on is specified) with `other`'s index, preserving the order           of the calling's one.     lsuffix : str, default ''         Suffix to use from left frame's overlapping columns.     rsuffix : str, default ''         Suffix to use from right frame's overlapping columns.     sort : bool, default False         Order result DataFrame lexicographically by the join key. If False,         the order of the join key depends on the join type (how keyword).          Returns     -------     DataFrame         A dataframe containing columns from both the caller and `other`.          See Also     --------     DataFrame.merge : For column(s)-on-column(s) operations.          Notes     -----     Parameters `on`, `lsuffix`, and `rsuffix` are not supported when     passing a list of `DataFrame` objects.          Support for specifying index levels as the `on` parameter was added     in version 0.23.0.          Examples     --------     >>> df = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],     ...                    'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})          >>> df       key   A     0  K0  A0     1  K1  A1     2  K2  A2     3  K3  A3     4  K4  A4     5  K5  A5          >>> other = pd.DataFrame({'key': ['K0', 'K1', 'K2'],     ...                       'B': ['B0', 'B1', 'B2']})          >>> other       key   B     0  K0  B0     1  K1  B1     2  K2  B2          Join DataFrames using their indexes.          >>> df.join(other, lsuffix='_caller', rsuffix='_other')       key_caller   A key_other    B     0         K0  A0        K0   B0     1         K1  A1        K1   B1     2         K2  A2        K2   B2     3         K3  A3       NaN  NaN     4         K4  A4       NaN  NaN     5         K5  A5       NaN  NaN          If we want to join using the key columns, we need to set key to be     the index in both `df` and `other`. The joined DataFrame will have     key as its index.          >>> df.set_index('key').join(other.set_index('key'))           A    B     key     K0   A0   B0     K1   A1   B1     K2   A2   B2     K3   A3  NaN     K4   A4  NaN     K5   A5  NaN          Another option to join using the key columns is to use the `on`     parameter. DataFrame
1: apply(func: 'AggFuncType', axis: 'Axis' = 0, raw: 'bool' = False, result_type=None, args=(), **kwargs)     Apply a function along an axis of the DataFrame.          Objects passed to the function are Series objects whose index is     either the DataFrame's index (``axis=0``) or the DataFrame's columns     (``axis=1``). By default (``result_type=None``), the final return type     is inferred from the return type of the applied function. Otherwise,     it depends on the `result_type` argument.          Parameters     ----------     func : function         Function to apply to each column or row.     axis : {0 or 'index', 1 or 'columns'}, default 0         Axis along which the function is applied:              * 0 or 'index': apply function to each column.         * 1 or 'columns': apply function to each row.          raw : bool, default False         Determines if row or column is passed as a Series or ndarray object:              * ``False`` : passes each row or column as a Series to the           function.         * ``True`` : the passed function will receive ndarray objects           instead.           If you are just applying a NumPy reduction function this will           achieve much better performance.          result_type : {'expand', 'reduce', 'broadcast', None}, default None         These only act when ``axis=1`` (columns):              * 'expand' : list-like results will be turned into columns.         * 'reduce' : returns a Series if possible rather than expanding           list-like results. This is the opposite of 'expand'.         * 'broadcast' : results will be broadcast to the original shape           of the DataFrame, the original index and columns will be           retained.              The default behaviour (None) depends on the return value of the         applied function: list-like results will be returned as a Series         of those. However if the apply function returns a Series these         are expanded to columns.     args : tuple         Positional arguments to pass to `func` in addition to the         array/series.     **kwargs         Additional keyword arguments to pass as keywords arguments to         `func`.          Returns     -------     Series or DataFrame         Result of applying ``func`` along the given axis of the         DataFrame.          See Also     --------     DataFrame.applymap: For elementwise operations.     DataFrame.aggregate: Only perform aggregating type operations.     DataFrame.transform: Only perform transforming type operations.          Notes     -----     Functions that mutate the passed object can produce unexpected     behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`     for more details.          Examples     --------     >>> df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])     >>> df        A  B     0  4  9     1  4  9     2  4  9          Using a numpy universal function (in this case the same as     ``np.sqrt(df)``):          >>> df.apply(np.sqrt)          A    B     0  2.0  3.0     1  2.0  3.0     2  2.0  3.0          Using a reducing function on either axis          >>> df.apply(np.sum, axis=0)     A    12     B    27     dtype: int64          >>> df.apply(np.sum, axis=1)     0    13     1    13     2    13     dtype: int64          Returning a list-like will result in a Series          >>> df.apply(lambda x: [1, 2], axis=1)     0    [1, 2]     1    [1, 2]     2    [1, 2]     dtype: object          Passing ``result_type='expand'`` will expand list-like results     to columns of a Dataframe          >>> df.apply(lambda x: [1, 2], axis=1, result_type='expand')        0  1     0  1  2     1  1  2     2  1  2          Returning a Series inside the function is similar to passing     ``result_type='expand'``. The resulting column names     will be the Series index.          >>> df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)        foo  bar     0    1    2     1    1    2     2    1    2          Passing ``result_type='broadcast'`` will ensure the same shape     result,
2: add_prefix(prefix: 'str')     Prefix labels with string `prefix`.          For Series, the row labels are prefixed.     For DataFrame, the column labels are prefixed.          Parameters     ----------     prefix : str         The string to add before each label.          Returns     -------     Series or DataFrame         New Series or DataFrame with updated labels.          See Also     --------     Series.add_suffix: Suffix row labels with string `suffix`.     DataFrame.add_suffix: Suffix column labels with string `suffix`.          Examples     --------     >>> s = pd.Series([1, 2, 3, 4])     >>> s     0    1     1    2     2    3     3    4     dtype: int64          >>> s.add_prefix('item_')     item_0    1     item_1    2     item_2    3     item_3    4     dtype: int64          >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})     >>> df        A  B     0  1  3     1  2  4     2  3  5     3  4  6          >>> df.add_prefix('col_')          col_A  col_B     0       1       3     1       2       4     2       3       5     3       4       6 
\n
## Problem: 
Sample dataframe: df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})  I'd like to add sigmoids of each existing column to the dataframe and name them based on existing column names with a prefix, e.g. sigmoid_A is an sigmoid of column A and so on. The resulting dataframe should look like so: result = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "sigmoid_A": [1/(1+e^(-1)), 1/(1+e^(-2)), 1/(1+e^(-3))], "sigmoid_B": [1/(1+e^(-4)), 1/(1+e^(-5)), 1/(1+e^(-6))]})  Notice that e is the natural constant. Obviously there are redundant methods like doing this in a loop, but there should exist much more pythonic ways of doing it and after searching for some time I didn't find anything. I understand that this is most probably a duplicate; if so, please point me to an existing answer.  
\n
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

"""
    answer3 = """
<code>
import math
def g(df):
    return df.join(df.apply(lambda x: 1/(1+math.e**(-x))).add_prefix('sigmoid_'))

result = g(df.copy())
</code>
"""

    user_prompt = f"""
## Potential documents:
{potential_docs}
\n
## Problem: 
{prompt}
\n
## Unfinished Code Solution:
{answer}
"""




    if model.startswith('llama2') or model.startswith('codellama'):
        prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n
{user_prompt} [/INST]
    """
    elif model.startswith('llama3'):
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{example1}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>{answer1}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{example2}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>{answer2}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{example3}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>{answer3}<|eot_id|>\n
<|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
<|start_header_id|>assistant<|end_header_id>
    """

    return prompt_template


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

    """
    ensemble  examples as shots
    """
    sampled_rest_id_type1 = ['Numpy_200', 'Tensorflow_25', 'Pandas_53', 'Numpy_41', 'Scipy_98', 'Scipy_66', 'Tensorflow_8', 'Numpy_96', 'Scipy_56', 'Numpy_166']
    sampled_rest_id_type2 = ['Matplotlib_93', 'Matplotlib_142', 'Matplotlib_145', 'Matplotlib_33', 'Matplotlib_80', 'Matplotlib_20', 'Matplotlib_44', 'Matplotlib_120', 'Matplotlib_14', 'Matplotlib_99']
    ds1000 = DS1000Dataset(source_dir=root_path + '/data/DS1000/ds1000_data', mode='Insertion', libs='all')
    shots = 4
    for sampled_id in sampled_rest_id_type1[:shots]:
        [lib, problem_id] = sampled_id.split('_')
        data = ds1000[lib][int(problem_id)]
        # print('qs_id', sampled_id)
        # print('reference code', data['reference_code'])
        # prompt, code = data['prompt'].split('A:')
        # print('prompt', prompt.replace('\n', ' '))
        # print('code', code)

    # api_signs = ['numpy.isclose', 'builtins.sum', 'tensorflow.gather_nd', 'pandas.core.frame.DataFrame.join', 'pandas.core.frame.DataFrame.apply', 'pandas.core.frame.DataFrame.add_prefix']
    # from prompt_utils import get_truncated_docs
    # get_truncated_docs(api_signs)

    """
    test prompt
    """
    ret_docs = ['asdadwasdawdsdawd', 'bcxvnmtgjr']
    question = ds1000['Pandas'][1]['prompt']
    print(llama_3shot_prompt(ret_docs, question, 'codellama-13b-instruct'))
