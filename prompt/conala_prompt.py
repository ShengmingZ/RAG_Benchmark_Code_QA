def llama_0shot_prompt(potential_docs, question):
    sys_prompt = """You are a senior python programmer, given some potential api documents starts with `## Potential documents` and a program description starts with `## Description`, 
you should first read the potential documents, and then write a python program according to the description in one line.
The program should starts with <code> and ends with </code>
"""

    user_prompt = f"""## Potential documents: 
    {potential_docs}
    ## Description: 
    {question}
    """

    prompt_template = f"""<s>[INST] <<SYS>>
    {sys_prompt} <</SYS>>\n
    {user_prompt} [/INST]
    """

    return prompt_template

def gpt_3shots_prompt(potential_docs, question):

    sys_prompt = """You are a senior python programmer, given some potential api documents starts with `## Potential documents` and a program description starts with `## Description`,
you should first read the potential documents, and then write a python program according to the description in one line, 
the program should starts with <code> and ends with </code>"""


    shots = """## Potential documents
0: outer(a, b, out=None)     Compute the outer product of two vectors.          Given two vectors, ``a = [a0, a1, ..., aM]`` and     ``b = [b0, b1, ..., bN]``,     the outer product [1]_ is::            [[a0*b0  a0*b1 ... a0*bN ]        [a1*b0    .        [ ...          .        [aM*b0            aM*bN ]]          Parameters     ----------     a : (M,) array_like         First input vector.  Input is flattened if         not already 1-dimensional.     b : (N,) array_like         Second input vector.  Input is flattened if         not already 1-dimensional.     out : (M, N) ndarray, optional         A location where the result is stored              .. versionadded:: 1.9.0          Returns     -------     out : (M, N) ndarray         ``out[i, j] = a[i] * b[j]``          See also     --------     inner     einsum : ``einsum('i,j->ij', a.ravel(), b.ravel())`` is the equivalent.     ufunc.outer : A generalization to dimensions other than 1D and other                   operations. ``np.multiply.outer(a.ravel(), b.ravel())``                   is the equivalent.     tensordot : ``np.tensordot(a.ravel(), b.ravel(), axes=((), ()))``                 is the equivalent.          References     ----------     .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd              ed., Baltimore, MD, Johns Hopkins University Press, 1996,              pg. 8.          Examples     --------     Make a (*very* coarse) grid for computing a Mandelbrot set:          >>> rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))     >>> rl     array([[-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.],            [-2., -1.,  0.,  1.,  2.]])     >>> im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))     >>> im     array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],            [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],            [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],            [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])     >>> grid = rl + im     >>> grid     array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],            [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],            [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],            [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],            [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])          An example using a "vector" of letters:          >>> x = np.array(['a', 'b', 'c'], dtype=object)     >>> np.outer(x, [1, 2, 3])     array([['a', 'aa', 'aaa'],            ['b', 'bb', 'bbb'],            ['c', 'cc', 'ccc']], dtype=object)  
## Description: 
multiplication of two 1-dimensional arrays  in numpy
## Answer: 
<code>np.outer(a, b)</code>


## Potential documents 
0: fromtimestamp()     timestamp[, tz] -> tz's local time from POSIX timestamp.  
1: strftime()     format -> strftime() style string.  
## Description:
convert epoch time represented as milliseconds `s` to string using format '%Y-%m-%d %H:%M:%S.%f'
## Answer:
<code>datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')</code>


## Potential documents:
0: loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, *, like=None)     Load data from a text file.          Each row in the text file must have the same number of values.          Parameters     ----------     fname : file, str, or pathlib.Path         File, filename, or generator to read.  If the filename extension is         ``.gz`` or ``.bz2``, the file is first decompressed. Note that         generators should return byte strings.     dtype : data-type, optional         Data-type of the resulting array; default: float.  If this is a         structured data-type, the resulting array will be 1-dimensional, and         each row will be interpreted as an element of the array.  In this         case, the number of columns used must match the number of fields in         the data-type.     comments : str or sequence of str, optional         The characters or list of characters used to indicate the start of a         comment. None implies no comments. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is '#'.     delimiter : str, optional         The string used to separate values. For backwards compatibility, byte         strings will be decoded as 'latin1'. The default is whitespace.     converters : dict, optional         A dictionary mapping column number to a function that will parse the         column string into the desired value.  E.g., if column 0 is a date         string: ``converters = {0: datestr2num}``.  Converters can also be         used to provide a default value for missing data (but see also         `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.         Default: None.     skiprows : int, optional         Skip the first `skiprows` lines, including comments; default: 0.     usecols : int or sequence, optional         Which columns to read, with 0 being the first. For example,         ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.         The default, None, results in all columns being read.              .. versionchanged:: 1.11.0             When a single column has to be read it is possible to use             an integer instead of a tuple. E.g ``usecols = 3`` reads the             fourth column the same way as ``usecols = (3,)`` would.     unpack : bool, optional         If True, the returned array is transposed, so that arguments may be         unpacked using ``x, y, z = loadtxt(...)``.  When used with a         structured data-type, arrays are returned for each field.         Default is False.     ndmin : int, optional         The returned array will have at least `ndmin` dimensions.         Otherwise mono-dimensional axes will be squeezed.         Legal values: 0 (default), 1 or 2.              .. versionadded:: 1.6.0     encoding : str, optional         Encoding used to decode the inputfile. Does not apply to input streams.         The special value 'bytes' enables backward compatibility workarounds         that ensures you receive byte arrays as results if possible and passes         'latin1' encoded strings to converters. Override this value to receive         unicode arrays and pass strings as input to converters.  If set to None         the system default is used. The default value is 'bytes'.              .. versionadded:: 1.14.0     max_rows : int, optional         Read `max_rows` lines of content after `skiprows` lines. The default         is to read all the lines.              .. versionadded:: 1.16.0     like : array_like         Reference object to allow the creation of arrays which are not         NumPy arrays. If an array-like passed in as ``like`` supports         the ``__array_function__`` protocol, the result will be defined         by it. In this case, it ensures the creation of an array object         compatible with that passed in via this argument.              .. versionadded:: 1.20.0          Returns     -------     out : ndarray         Data read from the text file.          See Also     --------     load, fromstring, fromregex     genfromtxt : Load data with missing values handled as specified.     scipy.io.loadmat : reads MATLAB data files          Notes     -----     This function aims to be a fast reader for simply formatted files.  The     `genfromtxt` function provides more sophisticated handling of, e.g.,     lines with missing values.          .. versionadded:: 1.10.0          The strings produced by the Python float.hex method can be used as     input for floats.          Examples     --------     >>> from io import StringIO   # StringIO behaves like a file object     >>> c = StringIO("0 1\n2 3")     >>> np.loadtxt(c)     array([[0., 1.],            [2., 3.]])          >>> d = StringIO("M 21 72\nF 35 58")     >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),     ...                      'formats': ('S1', 'i4', 'f4')})     array([(b'M', 21, 72.), (b'F', 35, 58.)],           dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])          >>> c = StringIO("1,0,2\n3,0,4")     >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)     >>> x     array([1., 3.])     >>> y     array([2., 4.])          This example shows how `converters` can be used to convert a field     with a trailing minus sign into a negative number.          >>> s = StringIO('10.01 31.25-\n19.22 64.31\n17.57- 63.94')     >>> def conv(fld):     ...     return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)     ...     >>> np.loadtxt(s, converters={0: conv, 1: conv})     array([[ 10.01, -31.25],            [ 19.22,  64.31],            [-17.57,  63.94]])  
## Description: 
convert csv file 'test.csv' into two-dimensional matrix
## Answer:
<code>numpy.loadtxt(open('test.csv', 'rb'), delimiter=',', skiprows=1)</code>
"""

    user_prompt = f"""## Potential documents: 
    {potential_docs}
    ## Description: 
    {question}
    # Answer:
    """

    return sys_prompt + '\n\n' + shots + '\n\n' + user_prompt


def llama_0shot_no_ret_prompt(question):
    sys_prompt = """You are a senior python programmer, given a program description starts with `## Description`, 
    you should write a python program according to the description in one line.
    The program should starts with <code> and ends with </code>
    """

    user_prompt = f"""## Description: 
    {question}
    """

    prompt_template = f"""<s>[INST] <<SYS>>
    {sys_prompt} <</SYS>>\n
    {user_prompt} [/INST]
    """

    return prompt_template


def gpt_3shots_no_ret_prompt():
    conala_original_no_retrieval_prompt = '''# convert string '2011221' into a DateTime object using format '%Y%W%w'
    datetime.strptime('2011221', '%Y%W%w')
    
    #END
    
    # Sort a list of strings 'words' such that items starting with 's' come first.
    sorted(words, key=lambda x: 'a' + x if x.startswith('s') else 'b' + x)
    
    #END
    
    # replace all the nan values with 0 in a pandas dataframe `df`
    df.fillna(0)
    
    #END
    '''

    conala_0shot_prompt = '''Given the description, and some potential documents that might help, generate corresponding Python command. 
    Only generate the command, and remember that some of the documents might not be helpful when generating the command.'''
    # And pay attention that potential documents might not be helpful when generating the command


if __name__ == '__main__':
    # def get_examples(num=3):
    #     import json, random
    #     data_file = '../data/conala/cmd_train.oracle_man.full.json'
    #     dataset = json.load(open(data_file, 'r'))
    #     examples = random.sample(dataset, num)
    #     for example in examples:
    #         print(example['nl'], example['cmd'])
    #
    # get_examples(3)

    from dataset_utils.corpus_utils import PythonDocsLoader
    api_signs = ['numpy.outer', 'datetime.datetime.fromtimestamp', 'datetime.datetime.strftime', 'numpy.loadtxt']
    docs = PythonDocsLoader().get_docs(api_signs)
    docs = [doc.replace('\n', ' ') for doc in docs]
    import tiktoken
    MAX_TOKENS = 2000
    encoding = tiktoken.get_encoding("cl100k_base")
    for doc in docs:
        encoded_doc = encoding.encode(doc)[:MAX_TOKENS]
        doc = encoding.decode(encoded_doc)
        print(doc)
