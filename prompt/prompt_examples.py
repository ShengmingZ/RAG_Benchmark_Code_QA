class PromptExample:
    NQ = None
    TriviaQA = None
    hotpotQA = None
    conala = None
    DS1000 = None
    pandas_numpy_eval = None

class zeroshot_with_instruct_gpt(PromptExample):
    def __init__(self):
        self.NQ = """You are a helpful assistant, given some potential documents starts with ## Potential documents and a question starts with ## Question,
you should first read the potential documents, and then use the knowledge in documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>

## Potential documents:
<PotentialDocuments>

## Question:
<Question>
"""
        self.TriviaQA = self.NQ
        self.hotpotQA = self.NQ
        self.conala = """You are a senior python programmer, given some potential api documents starts with ## Potential documents and a program description starts with ## Description,
you should first read the potential documents, and then write a python program according to the description in one line.
The program should starts with <code> and ends with </code>

## Potential documents: 
<PotentialDocuments>

## Description:
<Description>
"""
        self.DS1000 = """You are a senior python programmer, given some potential api documents starts with ## Potential documents, a program description starts with ## Problem, and the unfinished code solution starts with ## Unfinished Code Solution,
you should first read the potential documents, and then use the knowledge in documents to complete the code solution according to the problem.
you should only output the uncompleted part of the code solution, and the output code should start with <code> and end with </code>

## Potential documents: 
<PotentialDocuments>

## Problem:
<Problem>

## Unfinished Code Solution:
<UnfinishedCodeSolution>
"""
        self.pandas_numpy_eval = """You are a senior python programmer, given some potential api documents starts with ## Potential documents, and a unfinished code snippet starts with ## Unfinished Code Snippet,
you should first read the potential documents, and then use the knowledge in documents to complete the code snippet according to the comments in the code.
you should only output the uncompleted part of the code snippet, and the output code should starts with <code> and ends with </code>

## Potential documents:
<PotentialDocuments>

## Unfinished Code Snippet:
<UnfinishedCodeSnippet>
"""


class zeroshot_with_instruct_llama(PromptExample):
    def __init__(self):
        self.NQ = """<s>[INST] <<SYS>> You are a helpful assistant, given some potential documents starts with ## Potential documents and a question starts with ## Question,
you should first read the potential documents, and then use the knowledge in documents to answer the question.
You should only output the exact answer, and the answer should starts with <answer> and ends with </answer>
 <</SYS>>

## Potential documents:
<PotentialDocuments>

## Question:
<Question>

[/INST]
"""
        self.TriviaQA = self.NQ
        self.hotpotQA = self.NQ
        self.conala = """<s>[INST] <<SYS>> You are a senior python programmer, given some potential api documents starts with ## Potential documents and a program description starts with ## Description,
you should first read the potential documents, and then write a python program according to the description in one line.
The program should starts with <code> and ends with </code>
 <</SYS>>

## Potential documents: 
<PotentialDocuments>

## Description:
<Description>

[/INST]
"""
        self.DS1000 = """<s>[INST] <<SYS>> You are a senior python programmer, given some potential api documents starts with ## Potential documents, a program description starts with ## Problem, and the unfinished code solution starts with ## Unfinished Code Solution,
you should first read the potential documents, and then use the knowledge in documents to complete the code solution according to the problem.
you should only output the uncompleted part of the code solution, and the output code should start with <code> and end with </code>
 <</SYS>>

## Potential documents: 
<PotentialDocuments>

## Problem:
<Problem>

## Unfinished Code Solution:
<UnfinishedCodeSolution>

[/INST]
"""
        self.pandas_numpy_eval = """<s>[INST] <<SYS>> You are a senior python programmer, given some potential api documents starts with `## Potential documents`, and a unfinished code snippet starts with `## Unfinished Code Snippet`, 
you should first read the potential documents, and then use the knowledge in documents to complete the code snippet according to the comments in the code.
you should only output the uncompleted part of the code snippet, and the output code should starts with <code> and ends with </code>
 <</SYS>>
 
 ## Potential documents:
<PotentialDocuments>

## Unfinished Code Snippet:
<UnfinishedCodeSnippet>

[/INST]
"""


class fewshot_gpt(PromptExample):
    def __init__(self):
        self.NQ = ''''Follow the examples to solve the last question

## Potential documents:
0: of the 2nd Texas Mounted Rifles under Lieutenant Colonel John R. Baylor was sent to occupy the series of forts along the western Texas frontier which had been abandoned by the Union Army. Baylor's orders from the Department of Texas commander, Colonel Earl Van Dorn, allowed him to advance into New Mexico in order to attack the Union forts along the Rio Grande if he thought the situation called for such measures. Convinced that the Union force at Fort Fillmore would soon attack, Baylor decided to take the initiative and launch an attack of his own. Leaving during the night

## Question: 
who led the confederate force that captured fort fillmore

## Answer:
```Lieutenant Colonel John R. Baylor```



## Potential documents:
0: in the city and the sun shines on LA.' I didn't like the way it sounded at the time. And so I just had it sitting back in the corner. Then life changed my plans once again, and I was now facing joining Journey. I love San Francisco, the bay, and the whole thing. 'The bay' fit so nice, 'When the lights go down in the city and the sun shines on the bay.' It was one of those early-morning-going-across-the-bridge things, when the sun was coming up and the lights were going down. It was perfect."" Released as a single

## Question: 
who sings when the lights go down in the city

## Answer:
```Journey```



## Potential documents:
0: Prokaryote A prokaryote is usually a unicellular organism, sometimes a multi-cellular organism, that lacks a membrane-bound nucleus, mitochondria, or any other membrane-bound organelle. The word ""prokaryote"" comes from the Greek πρό (""pro"") ""before"" and κάρυον (""karyon"") ""nut or kernel"". Prokaryotes are divided into two domains, Archaea and Bacteria. In contrast, species with nuclei and organelles are placed in the third domain, Eukaryota. Prokaryotes reproduce without fusion of gametes. The first living organisms are thought to have been prokaryotes. In the prokaryotes, all the intracellular water-soluble components (proteins, DNA and metabolites) are located together in the cytoplasm enclosed by the cell

## Question: 
what type of cell has no nucleus or membrane bound organelles

## Answer:
```prokaryote```



## Potential documents:
<PotentialDocuments>

## Question: 
<Question>

# Answer:
'''
        self.TriviaQA = self.NQ
        self.hotpotQA = """Follow the examples to solve the last question

## Potential documents:
0: Danielle Prendergast (born September 8, 1990), better known by her stage name Elle Royal (formerly known as Patwa), is an independent Hip-Hop artist hailing from The Bronx, New York. Her breakthrough came in 2010 when her video "What Can I Say" went viral after WorldStarHipHop featured her as the “Female Artist of the Week”. Elle Royal later released the mixtape One Gyal Army under Patwa in 2010, followed by the singles “Jammin”, “Lights”, and “Statements” in 2015 under her current stage name, Elle Royal.
1: WorldStarHipHop is a content-aggregating video blog. Founded in 2005, the site averages 528,726 unique visitors a day. Alexa ranks the site 342nd in site traffic in the United States and 1,212th for worldwide traffic. The site, operated by Worldstar, LLC, was founded at age 33 by Lee "Q" O' Denat, a Hollis, Queens-based hip-hop fan and Grover Cleveland High School dropout. Described by "Vibe" as a "remnant of the Geocities generation," the site regularly features public fighting caught on video, music videos and assorted content targeted to young audiences. O'Denat refers to the site as the "CNN of the ghetto." In 2012, Alexa Internet stated "Compared with all Internet users, its users are disproportionately young people and they tend to be childless, moderately educated men 18–21 who browse from school and work."

## Question: 
Elle Royal's video "What Can I Say" went viral after she was featured as “Female Artist of the Week” by a video blog founded in what year?

## Answer:
```2005```



## Potential documents:
0: The 2003 LSU Tigers football team represented Louisiana State University (LSU) during the 2003 NCAA Division I-A football season. Coached by Nick Saban, the LSU Tigers played their home games at Tiger Stadium in Baton Rouge, Louisiana. The Tigers compiled an 11–1 regular season record and then defeated the No. 5 Georgia Bulldogs in the SEC Championship Game, Afterward, LSU was invited to play the Oklahoma Sooners in the Sugar Bowl for the Bowl Championship Series (BCS) national title. LSU won the BCS National Championship Game, the first national football championship for LSU since 1958.
1: The 2004 Nokia Sugar Bowl, the BCS title game for the 2003 college football season, was played on January 4, 2004 at the Louisiana Superdome in New Orleans, Louisiana. The teams were the LSU Tigers and the Oklahoma Sooners. The Tigers won the BCS National Championship, their second championship, defeating the Sooners by a score of 21-14.

## Question: 
What game did the team with an 11-1 regular season record play in for the BCS title game?

## Answer:
```2004 Nokia Sugar Bowl```



## Potential documents:
0: The 2011 Teen Choice Awards ceremony, hosted by Kaley Cuoco, aired live on August 7, 2011 at 8/7c on Fox. This was the first time that the ceremonies were aired live since the 2007 show.
1: Kaley Christine Cuoco ( ; born November 30, 1985) is an American actress. After a series of supporting film and television roles in the late 1990s, she landed her breakthrough role as Bridget Hennessy on the ABC sitcom "8 Simple Rules", on which she starred from 2002 to 2005. Thereafter, Cuoco appeared as Billie Jenkins on the final season of the television series "Charmed" (2005–2006). Since 2007, she has starred as Penny on the CBS sitcom "The Big Bang Theory", for which she has received Satellite, Critics' Choice, and People's Choice Awards. Cuoco's film work includes roles in "To Be Fat like Me" (2007), "Hop" (2011) and "Authors Anonymous" (2014). She received a star on the Hollywood Walk of Fame in 2014.

## Question: 
What show does the host of The 2011 Teen Choice Awards ceremony currently star on?

## Answer:
```The Big Bang Theory```



## Potential documents:
<PotentialDocuments>

## Question: 
<Question>

## Answer:
"""
        self.conala = """## Potential documents:
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



## Potential documents: 
<PotentialDocuments>

## Code Description: 
<CodeDescription>

## Generated Code:
"""
        self.DS1000 = """## Potential documents:
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



## Potential documents:
<PotentialDocuments>

## Problem: 
<Problem>

## Unfinished Code Solution:
<UnfinishedCodeSolution>

# [insert]:
"""
        self.pandas_numpy_eval = '''## Potential documents:
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

## Potential documents:
<PotentialDocuments>

## Unfinished Code Snippet:
<UnfinishedCodeSnippet>

## Completion:
'''


fewshot_llama = fewshot_gpt


class rar_gpt(PromptExample):
    def __init__(self):
        self.NQ = """You are a helpful assistant, given some Potential Documents and a Question,
Your task is to first rephrase and expand the question, then provide the answer

## Potential documents:
<PotentialDocuments>

## Question: 
<Question>
"""

        self.TriviaQA = self.NQ
        self.hotpotQA = self.NQ
        self.conala = """You are a senior python programmer, given some Potential Documents and a Code Description
Your task is to first rephrase and expand the Problem, then you should generate only one line Python statement tagged with ```

## Potential documents:
<PotentialDocuments>

## Code Description: 
<CodeDescription>

[/INST]
"""
        self.DS1000 = """You are a senior python programmer, given some potential documents, a problem and its unfinished code solution
Your task is to first rephrase and expand the Problem, then complete the Unfinished Code Solution without modifying its existing part

## Potential documents:
<PotentialDocuments>

## Problem: 
<Problem>

## Unfinished Code Solution:
<UnfinishedCodeSolution>
"""
        self.pandas_numpy_eval = """You are a senior python programmer, given some Potential Documents and a Uncompleted Code Snippet
Your task is to first rephrase and expand the Problem, then complete the code snippet

## Potential documents:
<PotentialDocuments>

## Uncompleted Code Snippet:
<UncompletedCodeSnippet>
"""




class rar_llama(PromptExample):
    def __init__(self):
        self.NQ = """
    """
        self.TriviaQA = """
    """
        self.hotpotQA = """
    """
        self.conala = """<s>[INST] <<SYS>> You are a senior python programmer, given some Potential Documents and a Code Description
Your task is to first rephrase and expand the Problem, then you should generate only one line Python statement tagged with ``` <</SYS>>

## Potential documents:
<PotentialDocuments>

## Code Description: 
<CodeDescription>

[/INST]
"""
        self.DS1000 = """<s>[INST] <<SYS>> You are a senior python programmer, given some potential documents, a problem and its unfinished code solution
Your task is to first rephrase and expand the Problem, then complete the Unfinished Code Solution without modifying its existing part <</SYS>>

## Potential documents:
<PotentialDocuments>

## Problem: 
<Problem>

## Unfinished Code Solution:
<UnfinishedCodeSolution>

[/INST]
"""
        self.pandas_numpy_eval = """<s>[INST] <<SYS>> You are a senior python programmer, given some Potential Documents and a Uncompleted Code Snippet
Your task is to first rephrase and expand the Problem, then complete the code snippet <</SYS>>

## Potential documents:
<PotentialDocuments>

## Uncompleted Code Snippet:
<UncompletedCodeSnippet>

[/INST]
"""


class cot_gpt(PromptExample):
    def __init__(self):
        self.NQ = """
"""
        self.TriviaQA = """
"""
        self.hotpotQA = """
"""
        self.conala = """
""",
        self.DS1000 = """
""",
        self.pandas_numpy_eval = """
"""


class cot_llama(PromptExample):
    def __init__(self):
        self.NQ = """
    """
        self.TriviaQA = """
    """
        self.hotpotQA = """
    """
        self.conala = """
    """,
        self.DS1000 = """
    """,
        self.pandas_numpy_eval = """
    """


self_consistency_gpt = cot_gpt
self_consistency_llama = cot_llama


class least_to_most_gpt(PromptExample):
    def __init__(self):
        self.NQ = """
    """
        self.TriviaQA = """
    """
        self.hotpotQA = """
    """
        self.conala = """
    """,
        self.DS1000 = """
    """,
        self.pandas_numpy_eval = """
    """

class least_to_most_llama(PromptExample):
    def __init__(self):
        self.NQ = """
    """
        self.TriviaQA = """
    """
        self.hotpotQA = """
    """
        self.conala = """
    """,
        self.DS1000 = """
    """,
        self.pandas_numpy_eval = """
    """


class plan_and_solve_gpt(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


class plan_and_solve_llama(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


class self_refine_gpt(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


class self_refine_llama(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


class con_gpt(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


class con_llama(PromptExample):
    def __init__(self):
        self.NQ = """
        """
        self.TriviaQA = """
        """
        self.hotpotQA = """
        """
        self.conala = """
        """,
        self.DS1000 = """
        """,
        self.pandas_numpy_eval = """
        """


ir_cot_gpt = cot_gpt
ir_cot_llama = cot_llama
flare_gpt = cot_gpt
flare_llama = cot_llama
