import sys
sys.path.append('/home/zhaoshengming/RAG_Benchmark_Code_QA')
from prompt.prompt_utils import ensemble_prompt


# LLAMA_SYS_PROMPT = """You are a senior python programmer, given some potentially useful api documentation tagged `## Potential documents` and a program description with uncompleted code tagged `## Description`, your task is to complete the python program.
# You should generate the complete python function, and the function should start with <code> and end with </code>
# """

LLAMA_SYS_PROMPT = """You are a senior python programmer. 

Input:
- useful api documents tagged `## API Documents`
- a program description with uncompleted code tagged `## Description`.

Task:
Follow the API documents and the program description to complete the python program.

Output Rules:
1. Generate the complete python function, keep existing code exactly the same
2. Only output the complete code in <code> and </code> tags
"""

# todo: new no ret:
LLAMA_SYS_PROMPT_NO_RET = """You are a senior python programmer.

Input:
- a program description with uncompleted code tagged `## Description`.

Task:
Follow the program description to complete the python program.

Output Rules:
1. Generate the complete python function, keep existing code exactly the same
2. Only output the complete code in <code> and </code> tags
"""


SYS_PROMPT_ZERO_SHOT = """Input:
- useful api documents tagged `## API Documents`
- a program description with uncompleted code tagged `## Description`.

Task:
Follow the program description to complete the python program.

Output Rules:
1. Generate the complete python function, keep existing code exactly the same
2. Only output the complete code, in <code> and </code> tags
"""


# # todo: OG no ret prompt:
# LLAMA_SYS_PROMPT_NO_RET = """You are a senior python programmer, given a program description with uncompleted code tagged `## Description`, your task is to complete the python program.
#  You should generate the complete python function, and the function should start with <code> and end with </code>
# """

# SYS_PROMPT_LEAST_TO_MOST = """Follow the examples to solve the last problem"""



def prompt_0shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Description: 
{question}
"""

#     if model.startswith('llama2') or model.startswith('codellama'):
#         prompt_template = f"""<s>[INST] <<SYS>> {LLAMA_SYS_PROMPT} <</SYS>>\n{user_prompt} [/INST]"""
#     elif model.startswith('llama3'):
#         prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{LLAMA_SYS_PROMPT}<|eot_id|>\n
# <|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
# <|start_header_id|>assistant<|end_header_id>
# """
#     elif model.startswith('gpt'):
#         prompt_template = LLAMA_SYS_PROMPT + '\n\n' + user_prompt
#     else:
#         raise ValueError(f'Unrecognized model: {model}')
    prompt_template = ensemble_prompt(sys_prompt=LLAMA_SYS_PROMPT, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_0shot_no_ret(question, model, pads=''):
    user_prompt = f"""
{pads}\n
## Description: 
{question}
"""
    sys_prompt = LLAMA_SYS_PROMPT_NO_RET

    # if model.startswith('llama2') or model.startswith('codellama'):
    #     prompt_template = f"""<s>[INST] <<SYS>> {sys_prompt} <</SYS>>\n{user_prompt} [/INST]"""
    # elif model.startswith('llama3'):
    #     prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{sys_prompt}<|eot_id|>\n
    # <|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|>\n
    # <|start_header_id|>assistant<|end_header_id>
    # """
    # elif model.startswith('gpt'):
    #     prompt_template = sys_prompt + '\n\n' + user_prompt
    # else:
    #     raise ValueError(f'Unrecognized model: {model}')
    prompt_template = ensemble_prompt(sys_prompt=sys_prompt, user_prompt=user_prompt, model=model)
    return prompt_template


def prompt_emotion(ret_docs, question, model):
    system_prompt_emotion = """This is very important to my career.

Input:
- useful api documents tagged `## API Documents`
- a program description with uncompleted code tagged `## Description`.

Task:
Follow the program description to complete the python program. 

Output Rules:
1. Generate the complete python function, keep existing code exactly the same
2. Only output the complete code, in <code> and </code> tags
"""
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Description: 
{question}
"""
    prompt_template = ensemble_prompt(sys_prompt=system_prompt_emotion, user_prompt=user_prompt, model=model)
    return prompt_template




def prompt_zero_shot_cot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Description: 
{question}

Let's think it step by step.
"""
    prompt_template = ensemble_prompt(sys_prompt=SYS_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template



def prompt_cot(ret_docs, question, model, existing_output=None):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    cot_prompt = """## API Documents:
0: outer(a, b, out=None)
    Compute the outer product of two vectors.
    
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    
      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]
    
    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.
    
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
             ed., Baltimore, MD, Johns Hopkins University Press, 1996,
             pg. 8.
    
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,
           
## Description: 
multiplication of two 1-dimensional arrays  in numpy

def f_23566515(a, b):\n\treturn 



According to the description, I need to multiply two 1-dimensional arrays in numpy.
The document shows outer(a, b) which computes the outer product of two vectors.
Since outer product is multiplication of 1D arrays, I should use np.outer(a, b).
Based on this, I implement the complete function while existing code exactly the same:
```
def f_23566515(a, b):
    return np.outer(a, b)
```




## API Documents:
0: fromtimestamp()
    timestamp[, tz] -> tz's local time from POSIX timestamp.
    
    
1: strftime()
    format -> strftime() style string. 



## Description: 
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'

def f_21787496(s):\n\treturn



According to the description, I need to convert epoch time in milliseconds to a formatted string.
The document shows fromtimestamp() which converts POSIX timestamp to local time.
The document also shows strftime() which formats time as a string.
Since I need to convert timestamp then format it, I should use datetime.fromtimestamp(s) then .strftime('%Y-%m-%d %H:%M:%S.%f').
Based on this, I implement the complete function while existing code exactly the same:
```
def f_21787496(s):
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
```



## API Documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)
    Load data from a text file.
    
    Each row in the text file must have the same number of values.
    
    Parameters
    ----------
    fname : file, str, or pathlib.Path
        File, filename, or generator_deprecated to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the
        column string into the desired value.  E.g., if column 0 is a date
        string: ``converters = {0: datestr2num}``.  Converters can also be
        used to provide a default value for missing data (but see also
        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
        Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The
        
        
## Description: 
convert csv file 'test.csv' into two-dimensional matrix

def f_4315506():\n\treturn



According to the description, I need to convert a CSV file 'test.csv' into a two-dimensional matrix.
The document shows loadtxt() which loads data from a text file into an array.
Since CSV files have comma-separated values, I need to set delimiter=',' parameter.
CSV files often have headers, so I should use skiprows=1 to skip the first row.
Based on this, I implement the complete function while existing code exactly the same:
```
def f_4315506():
    return numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
```
"""

    user_prompt = f"""
{cot_prompt}
\n\n
## API Documents: 
{potential_docs}
\n
## Description: 
{question}
"""

    if existing_output is not None: user_prompt = user_prompt + '\n' + existing_output

    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt



def prompt_3shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    examples_prompt = """## API Documents:
0: outer(a, b, out=None)
    Compute the outer product of two vectors.
    
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    
      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]
    
    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.
    
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
             ed., Baltimore, MD, Johns Hopkins University Press, 1996,
             pg. 8.
    
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,
           
## Description: 
multiplication of two 1-dimensional arrays  in numpy

def f_23566515(a, b):\n\treturn 


```
def f_23566515(a, b):
    return np.outer(a, b)
```



## API Documents
0: fromtimestamp()
    timestamp[, tz] -> tz's local time from POSIX timestamp.
    
    
1: strftime()
    format -> strftime() style string.
    
    

## Description: 
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'

def f_21787496(s):\n\treturn


```
def f_21787496(s):
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
```



## API Documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)
    Load data from a text file.
    
    Each row in the text file must have the same number of values.
    
    Parameters
    ----------
    fname : file, str, or pathlib.Path
        File, filename, or generator_deprecated to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the
        column string into the desired value.  E.g., if column 0 is a date
        string: ``converters = {0: datestr2num}``.  Converters can also be
        used to provide a default value for missing data (but see also
        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
        Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The
        
## Description: 
convert csv file 'test.csv' into two-dimensional matrix

def f_4315506():\n\treturn



```
def f_4315506():
    return numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
```
"""

    user_prompt = f"""
{examples_prompt}
\n\n
## API Documents: 
{potential_docs}
\n
## Description: 
{question}
"""

    # prompt_template = ensemble_prompt(sys_prompt=LLAMA_SYS_PROMPT,
    #                                   user_prompt=user_prompt,
    #                                   model=model,
    #                                   examples=[example1, example2, example3],
    #                                   answers=[answer1, answer2, answer3]
    #                                   )
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt


if __name__ == '__main__':
    example_doc = "loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)\n    Load data from a text file.\n    \n    Each row in the text file must have the same number of values.\n    \n    Parameters\n    ----------\n    fname : file, str, or pathlib.Path\n        File, filename, or generator_deprecated to read.  If the filename extension is\n        ``.gz`` or ``.bz2``, the file is first decompressed. Note that\n        generators should return byte strings.\n    dtype : data-type, optional\n        Data-type of the resulting array; default: float.  If this is a\n        structured data-type, the resulting array will be 1-dimensional, and\n        each row will be interpreted as an element of the array.  In this\n        case, the number of columns used must match the number of fields in\n        the data-type.\n    comments : str or sequence of str, optional\n        The characters or list of characters used to indicate the start of a\n        comment. None implies no comments. For backwards compatibility, byte\n        strings will be decoded as 'latin1'. The default is '#'.\n    delimiter : str, optional\n        The string used to separate values. For backwards compatibility, byte\n        strings will be decoded as 'latin1'. The default is whitespace.\n    converters : dict, optional\n        A dictionary mapping column number to a function that will parse the\n        column string into the desired value.  E.g., if column 0 is a date\n        string: ``converters = {0: datestr2num}``.  Converters can also be\n        used to provide a default value for missing data (but see also\n        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.\n        Default: None.\n    skiprows : int, optional\n        Skip the first `skiprows` lines, including comments; default: 0.\n    usecols : int or sequence, optional\n        Which columns to read, with 0 being the first. For example,\n        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.\n        The default, None, results in all columns being read.\n    \n        .. versionchanged:: 1.11.0\n            When a single column has to be read it is possible to use\n            an integer instead of a tuple. E.g ``usecols = 3`` reads the\n            fourth column the same way as ``usecols = (3,)`` would.\n    unpack : bool, optional\n        If True, the returned array is transposed, so that arguments may be\n        unpacked using ``x, y, z = loadtxt(...)``.  When used with a\n        structured data-type, arrays are returned for each field.\n        Default is False.\n    ndmin : int, optional\n        The returned array will have at least `ndmin` dimensions.\n        Otherwise mono-dimensional axes will be squeezed.\n        Legal values: 0 (default), 1 or 2.\n    \n        .. versionadded:: 1.6.0\n    encoding : str, optional\n        Encoding used to decode the inputfile. Does not apply to input streams.\n        The special value 'bytes' enables backward compatibility workarounds\n        that ensures you receive byte arrays as results if possible and passes\n        'latin1' encoded strings to converters. Override this value to receive\n        unicode arrays and pass strings as input to converters.  If set to None\n        the system default is used. The default value is 'bytes'.\n    \n        .. versionadded:: 1.14.0\n    max_rows : int, optional\n        Read `max_rows` lines of content after `skiprows` lines. The default\n        is to read all the lines.\n    \n        .. versionadded:: 1.16.0\n    like : array_like\n        Reference object to allow the creation of arrays which are not\n        NumPy arrays. If an array-like passed in as ``like`` supports\n        the ``__array_function__`` protocol, the result will be defined\n        by it. In this case, it ensures the creation of an array object\n        compatible with that passed in via this argument.\n    \n        .. versionadded:: 1.20.0\n    \n    Returns\n    -------\n    out : ndarray\n        Data read from the text file.\n    \n    See Also\n    --------\n    load, fromstring, fromregex\n    genfromtxt : Load data with missing values handled as specified.\n    scipy.io.loadmat : reads MATLAB data files\n    \n    Notes\n    -----\n    This function aims to be a fast reader for simply formatted files.  The\n    `genfromtxt` function provides more sophisticated handling of, e.g.,\n    lines with missing values.\n    \n    .. versionadded:: 1.10.0\n    \n    The strings produced by the Python float.hex method can be used as\n    input for floats.\n    \n    Examples\n    --------\n    >>> from io import StringIO   # StringIO behaves like a file object\n    >>> c = StringIO(\"0 1\\n2 3\")\n    >>> np.loadtxt(c)\n    array([[0., 1.],\n           [2., 3.]])\n    \n    >>> d = StringIO(\"M 21 72\\nF 35 58\")\n    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),\n    ...                      'formats': ('S1', 'i4', 'f4')})\n    array([(b'M', 21, 72.), (b'F', 35, 58.)],\n          dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])\n    \n    >>> c = StringIO(\"1,0,2\\n3,0,4\")\n    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)\n    >>> x\n    array([1., 3.])\n    >>> y\n    array([2., 4.])\n    \n    This example shows how `converters` can be used to convert a field\n    with a trailing minus sign into a negative number.\n    \n    >>> s = StringIO('10.01 31.25-\\n19.22 64.31\\n17.57- 63.94')\n    >>> def conv(fld):\n    ...     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)\n    ...\n    >>> np.loadtxt(s, converters={0: conv, 1: conv})\n    array([[ 10.01, -31.25],\n           [ 19.22,  64.31],\n           [-17.57,  63.94]])\n\n"
    from generator_deprecated.generate_utils import truncate_docs
    print(truncate_docs([example_doc], 'gpt-3.5-turbo-0125', 500)[0])






def prompt_least_to_most(ret_docs, question, model):
    examples_least_to_most = """## API Documents:
0: outer(a, b, out=None)
    Compute the outer product of two vectors.
    
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    
      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]
    
    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.
    
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
             ed., Baltimore, MD, Johns Hopkins University Press, 1996,
             pg. 8.
    
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,
           
## Description: 
multiplication of two 1-dimensional arrays  in numpy

def f_23566515(a, b):\n\treturn 
    
    
    
Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: What is the main task?
I need to perform multiplication of two 1-dimensional arrays in numpy.

Sub problem 2: Which API function can multiply 1D arrays?
From the documentation, outer(a, b) computes the outer product of two vectors, which is a form of multiplication between 1D arrays.

Now combining all solutions to generate the complete code while keeping existing code exactly the same:
```
def f_23566515(a, b):
    return np.outer(a, b)
```




## API Documents
0: fromtimestamp()
    timestamp[, tz] -> tz's local time from POSIX timestamp.
    
    
1: strftime()
    format -> strftime() style string.
    
    

## Description: 
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'

def f_21787496(s):\n\treturn



Let me break this down into simpler sub problems and solve them step by step:

Sub problem 1: What is the main task?
I need to convert epoch time in milliseconds to a formatted string.

Sub problem 2: Which API functions can help with time conversion?
From the documentation, fromtimestamp() converts POSIX timestamp to local time, and strftime() formats time as a string.

Sub problem 3: How do I chain these functions together?
I need to first convert the timestamp using fromtimestamp(s), then format it using strftime() with the given format '%Y-%m-%d %H:%M:%S.%f'.

Now combining all solutions to generate the complete code while keeping existing code exactly the same:
```
def f_21787496(s):
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
```




## API Documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)
    Load data from a text file.
    
    Each row in the text file must have the same number of values.
    
    Parameters
    ----------
    fname : file, str, or pathlib.Path
        File, filename, or generator_deprecated to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the
        column string into the desired value.  E.g., if column 0 is a date
        string: ``converters = {0: datestr2num}``.  Converters can also be
        used to provide a default value for missing data (but see also
        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
        Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The
        
## Description: 
convert csv file 'test.csv' into two-dimensional matrix

def f_4315506():\n\treturn


Let me break this down into simpler subproblems and solve them step by step:

Sub problem 1: What is the main task?
I need to convert a CSV file into a two-dimensional matrix.

Sub problem 2: Which API function can load data from text files?
From the documentation, loadtxt() can load data from a text file.

Sub problem 3: What parameters do I need for CSV files?
For CSV files, I need to set delimiter=',' to handle comma-separated values.
CSV files often have headers, so I should use skiprows=1 to skip the first row.

Now combining all solutions to generate the complete code while keeping existing code exactly the same:
```
def f_4315506():
    return numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
```
"""

    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""
{examples_least_to_most}
\n
## API Documents:
{potential_docs}
\n
## Description: 
{question}
"""
    # sys_prompt = 'You should generate the code in one line'
    # prompt = ensemble_prompt(sys_prompt, user_prompt, model)
    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt


def prompt_plan_and_solve(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'

    user_prompt = f"""## API Documents: 
{potential_docs}
\n
## Description: 
{question}
"""
    prompt_template = ensemble_prompt(sys_prompt=SYS_PROMPT_ZERO_SHOT, user_prompt=user_prompt, model=model)
    return prompt_template



# def prompt_ape():
# #     eval_prompt = """Instruction: You are a senior python programmer, given a Input Output pair,
# # where the Input is a program description augmented with some potential api documents, and the Output is corresponding generated code.
# # Your task is to generate the quality of the Output"""
#
#     eval_template = """Instruction: [PROMPT]
# Input: [INPUT]
# Output: [OUTPUT]"""
#
#     prompts = [""""## Potential documents:
# 0: outer(a, b, out=None)     Compute the outer product of two vectors.          Given two vectors, ``a = [a0, a1, ..., aM]`` and     ``b = [b0, b1, ..., bN]``,     the outer product [1]_ is::            [[a0*b0  a0*b1 ... a0*bN ]        [a1*b0    .        [ ...          .        [aM*b0            aM*bN ]]          Parameters     ----------     a : (M,) array_like         First input vector.  Input is flattened if         not already 1-dimensional.     b : (N,) array_like         Second input vector.  Input is flattened if         not already 1-dimensional.     out : (M, N) ndarray, optional         A location where the result is stored              .. versionadded:: 1.9.0          Returns     -------     out : (M, N) ndarray         ``out[i, j] = a[i] * b[j]``          See also     --------     inner     einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.     ufunc.outer : A generalization to dimensions other than 1D and other                   operations. ``np.multiply.outer(a.ravel(), b.ravel())``                   is the equivalent.     tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``                 is the equivalent.          References     ----------     .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd              ed., Baltimore, MD, Johns Hopkins University Press, 1996,              pg. 8.          Examples     --------     Make a (*very* coarse) grid for computing a Mandelbrot set:          >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))     >>> rl     array([[-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.]])     >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))     >>> im     array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],            [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],            [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],            [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])     >>> grid = rl + im     >>> grid     array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],            [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],            [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],            [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],            [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])          An example using a "vector" of letters:          >>> x = np.array(['a', 'b', 'c'], dtype=object)     >>> np.outer(x, [1, 2, 3])     array([['a', 'aa', 'aaa'],            ['b', 'bb', 'bbb'],            ['c', 'cc', 'ccc']], dtype=object)
#
# ## Code Description:
# multiplication of two 1-dimensional arrays  in numpy
#
# ## Generated Code:
# """,
#
# """"## Potential documents
# 0: fromtimestamp()     timestamp[, tz] -> tz's local time from POSIX timestamp.
# 1: strftime()     format -> strftime() style string.
#
# ## Code Description:
# convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'
#
# ## Generated Code:
# """,
#
# """## Potential documents:
# 0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)     Load data from a text file.          Each row in the text file must have the same number of values.          Parameters     ----------     fname : file, str, or pathlib.Path         File, filename, or generator_deprecated to read.  If the filename extension is         ``.gz`` or ``.bz2``, the file is first decompressed. Note that         generators should return byte strings.     dtype : data-type, optional         Data-type of the resulting array; default: float.  If this is a         structured data-type, the resulting array will be 1-dimensional, and         each row will be interpreted as an element of the array.  In this         case, the number of columns used must match the number of fields in         the data-type.     comments : str or sequence of str, optional         The characters or list of characters used to indicate the start of a         comment. None implies no comments. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is '#'.     delimiter : str, optional         The string used to separate values. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is whitespace.     converters : dict, optional         A dictionary mapping column number to a function that will parse the         column string into the desired value.  E.g., if column 0 is a date         string: ``converters = {0: datestr2num}``.  Converters can also be         used to provide a default value for missing data (but see also         `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.         Default: None.     skiprows : int, optional         Skip the first `skiprows` lines, including comments; default: 0.     usecols : int or sequence, optional         Which columns to read, with 0 being the first. For example,         ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.         The default, None, results in all columns being read.              .. versionchanged:: 1.11.0             When a single column has to be read it is possible to use             an integer instead of a tuple. E.g ``usecols = 3`` reads the             fourth column the same way as ``usecols = (3,)`` would.     unpack : bool, optional         If True, the returned array is transposed, so that arguments may be         unpacked using ``x, y, z = loadtxt(...)``.  When used with a         structured data-type, arrays are returned for each field.         Default is False.     ndmin : int, optional         The returned array will have at least `ndmin` dimensions.         Otherwise mono-dimensional axes will be squeezed.         Legal values: 0 (default), 1 or 2.              .. versionadded:: 1.6.0     encoding : str, optional         Encoding used to decode the inputfile. Does not apply to input streams.         The special value 'bytes' enables backward compatibility workarounds         that ensures you receive byte arrays as results if possible and passes         'latin1' encoded strings to converters. Override this value to receive         unicode arrays and pass strings as input to converters.  If set to None         the system default is used. The default value is 'bytes'.              .. versionadded:: 1.14.0     max_rows : int, optional         Read `max_rows` lines of content after `skiprows` lines. The default         is to read all the lines.              .. versionadded:: 1.16.0     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     out : ndarray         Data read from the text file.          See Also     --------     load, fromstring, fromregex     genfromtxt : Load data with missing values handled as specified.     scipy.io.loadmat : reads MATLAB data files          Notes     -----     This function aims to be a fast reader for simply formatted files.  The     `genfromtxt` function provides more sophisticated handling of, e.g.,     lines with missing values.          .. versionadded:: 1
#
# ## Code Description:
# convert csv file 'test.csv' into two-dimensional matrix
#
# ## Generated Code:
# """]
#
#     outputs = ["""```
# np.outer(a, b)
# ```
# """,
#
# """
# ```
# datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
# ```
# """,
#
# """
# ```
# numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
# ```
# """]
#
#     from automatic_prompt_engineer import ape
#
#     result, demo_fn = ape.simple_ape(dataset=(prompts, outputs), eval_template=eval_template)
#
#     print(result)
#     print(demo_fn)

# def prompt_RaR(ret_docs, question, model):
#     RaR_prompt = """You are a senior python programmer, given some Potential Documents and a Code Description
# Your task is to first rephrase and expand the Problem, then you should generate only one line Python statement tagged with ```"""
#
#     potential_docs = ''
#     for idx, ret_doc in enumerate(ret_docs):
#         potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'
#     user_prompt = f"""
# ## Potential documents:
# {potential_docs}
#
# ## Code Description:
# {question}
# """
#     sys_prompt = RaR_prompt
#     prompt = ensemble_prompt(sys_prompt, user_prompt, model)
#     # prompt = ['', user_prompt] if 'gpt' in model else user_prompt
#     return prompt


def prompt_self_refine(ret_docs, question, model, initial_output):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'

    self_refine_prompt = """## API Documents:
0: outer(a, b, out=None)
    Compute the outer product of two vectors.
    
    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::
    
      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]
    
    Parameters
    ----------
    a : (M,) array_like
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) array_like
        Second input vector.  Input is flattened if
        not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored
    
        .. versionadded:: 1.9.0
    
    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``
    
    See also
    --------
    inner
    einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.
    ufunc.outer : A generalization to dimensions other than 1D and other
                  operations. ``np.multiply.outer(a.ravel(), b.ravel())``
                  is the equivalent.
    tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``
                is the equivalent.
    
    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
             ed., Baltimore, MD, Johns Hopkins University Press, 1996,
             pg. 8.
    
    Examples
    --------
    Make a (*very* coarse) grid for computing a Mandelbrot set:
    
    >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,
           
## Description: 
multiplication of two 1-dimensional arrays  in numpy

def f_23566515(a, b):\n\treturn 


## Initial Code Solution:
```
def f_23566515(a, b):
    return a * b
```

Please first provide feedback on this solution and then refine it based on the feedback.

Feedback: The current solution uses element-wise multiplication (a * b), which only works if arrays have the same shape and produces a 1D result.
However, the task requires multiplication of two 1-dimensional arrays to create an outer product, which should produce a 2D matrix.

Refined solution:
```
def f_23566515(a, b):
    return np.outer(a, b)
```





## API Documents
0: fromtimestamp()
    timestamp[, tz] -> tz's local time from POSIX timestamp.
    
    
1: strftime()
    format -> strftime() style string.
    
    

## Description: 
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'

def f_21787496(s):\n\treturn


## Initial Code Solution:
```
def f_21787496(s):
    return s.strptime('%Y-%m-%d %H:%M:%S.%f')
```

Please first provide feedback on this solution and then refine it based on the feedback.

Feedback: The current solution uses strptime() which parses a string into a datetime object, but `s` is a numeric timestamp, not a string.
However, the task requires converting epoch time (numeric) to a formatted string, which needs fromtimestamp() to convert the number to datetime.

Refined solution:
```
def f_21787496(s):
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
```



## API Documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)
    Load data from a text file.
    
    Each row in the text file must have the same number of values.
    
    Parameters
    ----------
    fname : file, str, or pathlib.Path
        File, filename, or generator_deprecated to read.  If the filename extension is
        ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators should return byte strings.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence of str, optional
        The characters or list of characters used to indicate the start of a
        comment. None implies no comments. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is '#'.
    delimiter : str, optional
        The string used to separate values. For backwards compatibility, byte
        strings will be decoded as 'latin1'. The default is whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will parse the
        column string into the desired value.  E.g., if column 0 is a date
        string: ``converters = {0: datestr2num}``.  Converters can also be
        used to provide a default value for missing data (but see also
        `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
        Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines, including comments; default: 0.
    usecols : int or sequence, optional
        Which columns to read, with 0 being the first. For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The
        
## Description: 
convert csv file 'test.csv' into two-dimensional matrix

def f_4315506():\n\treturn


## Initial Code Solution:
```
def f_4315506():
    return numpy.loadtxt('test.csv')
```

Please first provide feedback on this solution and then refine it based on the feedback.

Feedback: The current solution doesn't specify the delimiter, so it will try to use whitespace as the default delimiter instead of commas.
However, the task requires loading a CSV file, which needs delimiter=',' to properly parse comma-separated values, and likely skiprows=1 to handle headers.

Refined solution:
```
def f_4315506():
    return numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
```
"""

    initial_output = f"```\n{initial_output}\n```"

    user_prompt = f"""
{self_refine_prompt}
\n\n\n
## API Documents: 
{potential_docs}
\n
## Description: 
{question}

## Initial Code Solution:
{initial_output}

Please first provide feedback on this solution and then refine it based on the feedback.
"""

    if 'gpt' in model: prompt = [dict(role='user', content=user_prompt)]
    else: prompt = f"<s>[INST] {user_prompt} [/INST]"
    return prompt



def prompt_con(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc + '\n\n'
    user_prompt = f"""## API Documents:
{potential_docs}
\n
## Description: 
{question}
"""
    sys_prompt_con = """Input:
- Useful API documents tagged `## API Documents`
- A program description with uncompleted code tagged `## Description`

Task:
Follow the program description to complete the python program by:
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
    prompt = ensemble_prompt(sys_prompt_con, user_prompt, model)
    return prompt



# if __name__ == '__main__':
#     # get examples
#     # def get_examples(num=3):
#     #     import json, random
#     #     data_file = '../data/conala/cmd_train.oracle_man.full.json'
#     #     dataset = json.load(open(data_file, 'r'))
#     #     examples = random.sample(dataset, num)
#     #     for example in examples:
#     #         print(example['nl'], example['cmd'])
#     #
#     # get_examples(3)
#
#     # get docs
#     # from prompt_utils import get_truncated_docs
#     # api_signs = ['numpy.outer', 'datetime.datetime.fromtimestamp', 'datetime.datetime.strftime', 'numpy.loadtxt']
#     # get_truncated_docs(api_signs)
#
#     prompt_ape()