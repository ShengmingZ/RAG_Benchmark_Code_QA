from prompt.prompt_utils import ensemble_prompt


LLAMA_SYS_PROMPT = """You are a senior python programmer, given some potential api documents starts with `## Potential documents` and a program description starts with `## Description`, 
you should first read the potential documents, and then write a python program according to the description in one line.
The program should starts with <code> and ends with </code>
"""

LLAMA_SYS_PROMPT_NO_RET = """You are a senior python programmer, given a program description starts with `## Description`, 
you need to write a python program according to the description in one line.
The program should starts with <code> and ends with </code>
"""

def prompt_3shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'

    examples_prompt = """
## Potential documents:
0: outer(a, b, out=None)     Compute the outer product of two vectors.          Given two vectors, ``a = [a0, a1, ..., aM]`` and     ``b = [b0, b1, ..., bN]``,     the outer product [1]_ is::            [[a0*b0  a0*b1 ... a0*bN ]        [a1*b0    .        [ ...          .        [aM*b0            aM*bN ]]          Parameters     ----------     a : (M,) array_like         First input vector.  Input is flattened if         not already 1-dimensional.     b : (N,) array_like         Second input vector.  Input is flattened if         not already 1-dimensional.     out : (M, N) ndarray, optional         A location where the result is stored              .. versionadded:: 1.9.0          Returns     -------     out : (M, N) ndarray         ``out[i, j] = a[i] * b[j]``          See also     --------     inner     einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.     ufunc.outer : A generalization to dimensions other than 1D and other                   operations. ``np.multiply.outer(a.ravel(), b.ravel())``                   is the equivalent.     tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``                 is the equivalent.          References     ----------     .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd              ed., Baltimore, MD, Johns Hopkins University Press, 1996,              pg. 8.          Examples     --------     Make a (*very* coarse) grid for computing a Mandelbrot set:          >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))     >>> rl     array([[-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.]])     >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))     >>> im     array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],            [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],            [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],            [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])     >>> grid = rl + im     >>> grid     array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],            [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],            [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],            [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],            [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])          An example using a "vector" of letters:          >>> x = np.array(['a', 'b', 'c'], dtype=object)     >>> np.outer(x, [1, 2, 3])     array([['a', 'aa', 'aaa'],            ['b', 'bb', 'bbb'],            ['c', 'cc', 'ccc']], dtype=object)  

## Code Description: 
multiplication of two 1-dimensional arrays  in numpy

## Generated Code:
```
np.outer(a, b)
```



## Potential documents
0: fromtimestamp()     timestamp[, tz] -> tz's local time from POSIX timestamp.  
1: strftime()     format -> strftime() style string.  

## Code Description: 
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'

## Generated Code:
```
datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
```



## Potential documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)     Load data from a text file.          Each row in the text file must have the same number of values.          Parameters     ----------     fname : file, str, or pathlib.Path         File, filename, or generator to read.  If the filename extension is         ``.gz`` or ``.bz2``, the file is first decompressed. Note that         generators should return byte strings.     dtype : data-type, optional         Data-type of the resulting array; default: float.  If this is a         structured data-type, the resulting array will be 1-dimensional, and         each row will be interpreted as an element of the array.  In this         case, the number of columns used must match the number of fields in         the data-type.     comments : str or sequence of str, optional         The characters or list of characters used to indicate the start of a         comment. None implies no comments. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is '#'.     delimiter : str, optional         The string used to separate values. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is whitespace.     converters : dict, optional         A dictionary mapping column number to a function that will parse the         column string into the desired value.  E.g., if column 0 is a date         string: ``converters = {0: datestr2num}``.  Converters can also be         used to provide a default value for missing data (but see also         `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.         Default: None.     skiprows : int, optional         Skip the first `skiprows` lines, including comments; default: 0.     usecols : int or sequence, optional         Which columns to read, with 0 being the first. For example,         ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.         The default, None, results in all columns being read.              .. versionchanged:: 1.11.0             When a single column has to be read it is possible to use             an integer instead of a tuple. E.g ``usecols = 3`` reads the             fourth column the same way as ``usecols = (3,)`` would.     unpack : bool, optional         If True, the returned array is transposed, so that arguments may be         unpacked using ``x, y, z = loadtxt(...)``.  When used with a         structured data-type, arrays are returned for each field.         Default is False.     ndmin : int, optional         The returned array will have at least `ndmin` dimensions.         Otherwise mono-dimensional axes will be squeezed.         Legal values: 0 (default), 1 or 2.              .. versionadded:: 1.6.0     encoding : str, optional         Encoding used to decode the inputfile. Does not apply to input streams.         The special value 'bytes' enables backward compatibility workarounds         that ensures you receive byte arrays as results if possible and passes         'latin1' encoded strings to converters. Override this value to receive         unicode arrays and pass strings as input to converters.  If set to None         the system default is used. The default value is 'bytes'.              .. versionadded:: 1.14.0     max_rows : int, optional         Read `max_rows` lines of content after `skiprows` lines. The default         is to read all the lines.              .. versionadded:: 1.16.0     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     out : ndarray         Data read from the text file.          See Also     --------     load, fromstring, fromregex     genfromtxt : Load data with missing values handled as specified.     scipy.io.loadmat : reads MATLAB data files          Notes     -----     This function aims to be a fast reader for simply formatted files.  The     `genfromtxt` function provides more sophisticated handling of, e.g.,     lines with missing values.          .. versionadded:: 1

## Code Description: 
convert csv file 'test.csv' into two-dimensional matrix

## Generated Code:
```
numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)
```
"""

    user_prompt = f"""
{examples_prompt}
\n
## Potential documents: 
{potential_docs}
## Code Description: 
{question}

## Generated Code:
"""

    # prompt_template = ensemble_prompt(sys_prompt=LLAMA_SYS_PROMPT,
    #                                   user_prompt=user_prompt,
    #                                   model=model,
    #                                   examples=[example1, example2, example3],
    #                                   answers=[answer1, answer2, answer3]
    #                                   )
    if 'gpt' in model: prompt = ['', user_prompt]
    else: prompt = user_prompt
    return prompt


def prompt_0shot(ret_docs, question, model):
    potential_docs = ''
    for idx, ret_doc in enumerate(ret_docs):
        potential_docs = potential_docs + f'{idx}: ' + ret_doc.replace('\n', ' ') + '\n'

    user_prompt = f"""## Potential documents: 
{potential_docs}
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


if __name__ == '__main__':
    # get examples
    # def get_examples(num=3):
    #     import json, random
    #     data_file = '../data/conala/cmd_train.oracle_man.full.json'
    #     dataset = json.load(open(data_file, 'r'))
    #     examples = random.sample(dataset, num)
    #     for example in examples:
    #         print(example['nl'], example['cmd'])
    #
    # get_examples(3)

    # get docs
    from prompt_utils import get_truncated_docs
    api_signs = ['numpy.outer', 'datetime.datetime.fromtimestamp', 'datetime.datetime.strftime', 'numpy.loadtxt']
    get_truncated_docs(api_signs)
