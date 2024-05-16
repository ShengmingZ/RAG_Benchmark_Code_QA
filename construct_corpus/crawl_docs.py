import inspect, pkgutil, importlib
import io, os, sys, json

# common used data science lib
third_party_lib_list = ['tensorflow', 'matplotlib', 'sklearn', 'numpy', 'torch', 'pandas', 'scipy', 'seaborn']

# python default libs from https://docs.python.org/3.7/library/index.html
py_builtin_lib_list = [
    'builtins',
    'string', 're', 'difflib', 'textwrap', 'unicodedata', 'stringprep', 'readline', 'rlcompleter',
    'struct', 'codecs',
    'datetime', 'calendar', 'collections', 'collections.abc', 'heapq', 'bisect', 'array', 'weakref', 'types', 'copy', 'pprint', 'reprlib', 'enum',
    'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random', 'statistics',
    'itertools', 'functools', 'operator',
    'pathlib', 'os.path', 'fileinput', 'stat', 'filecmp', 'tempfile', 'glob', 'fnmatch', 'linecache', 'shutil', 'macpath',
    'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
    'csv', 'configparser', 'netrc', 'xdrlib', 'plistlib',
    'hashlib', 'hmac', 'secrets',
    'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'logging.config', 'logging.handlers', 'getpass', 'curses', 'curses.textpad', 'curses.ascii', 'curses.panel', 'platform', 'errno', 'ctypes',
    'threading', 'multiprocessing', 'concurrent.futures', 'subprocess', 'sched', 'queue', '_thread', '_dummy_thread', 'dummy_threading',
    'contextvars', 'asyncio',
    'asyncio', 'socket', 'ssl', 'select', 'selectors', 'asyncore', 'asynchat', 'signal', 'mmap',
    'email', 'json', 'mailcap', 'mailbox', 'mimetypes', 'base64', 'binhex', 'binascii', 'quopri', 'uu',
    'html', 'html.parser', 'html.entities', 'xml',
    'webbrowser', 'cgi', 'cgitb', 'wsgiref', 'urllib', 'urllib.request', 'urllib.response', 'urllib.parse', 'urllib.error', 'urllib.robotparser', 'http', 'http.client', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'smtpd', 'telnetlib', 'uuid', 'socketserver', 'http.server', 'http.cookies', 'http.cookiejar', 'xmlrpc', 'xmlrpc.client', 'xmlrpc.server', 'ipaddress',
    'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr', 'ossaudiodev',
    'gettext', 'locale',
    'turtle', 'cmd', 'shlex',
    'tkinter', 'tkinter.ttk', 'tkinter.tix', 'tkinter.scrolledtext',
    'typing', 'pydoc', 'doctest', 'unittest', 'unittest.mock', 'lib2to3', 'test', 'test.support', 'test.support.script_helper',
    'bdb', 'faulthandler', 'pdb', 'cProfile', 'profile', 'timeit', 'trace', 'tracemalloc',
    'distutils', 'ensurepip', 'venv', 'zipapp',
    'sys', 'sysconfig', 'builtins', '__main__', 'warnings', 'dataclasses', 'contextlib', 'abc', 'atexit', 'traceback', '__future__', 'gc', 'inspect', 'site',
    'code', 'codeop',
    'zipimport', 'pkgutil', 'modulefinder', 'runpy', 'importlib',
    'parser', 'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize', 'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
    # formatter,
    # msilib, msvcrtm, winreg, winsound
    # posix, pwd, spwd, grp, crypt, termios, tty, pty, fcntl, pipes, resource, nis, syslog,
    # optparse, imp
]


def get_doc(attr_obj, full_name):
    if full_name == 'pydoc.help': return None
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        help(attr_obj)
        doc = buffer.getvalue()
    except:
        sys.stdout = sys.__stdout__
        print('exception for get doc on: ', full_name)
        doc = None
    finally:
        sys.stdout = sys.__stdout__

    return doc


def crawl_callable_attributes(module, library_name):
    module_name = module.__module__ + '.' + module.__name__ if hasattr(module, '__module__') else module.__name__
    if not module_name.startswith(library_name): return  # filter module not belong to this lib
    if module in module_list: return    # filter already traversed module
    module_list.append(module)

    if not hasattr(module, '__path__'): # if the module is not a package
        for item in inspect.getmembers(module):
            attr_name = item[0]
            if attr_name.startswith('__'): continue
            full_name = module_name + '.' + attr_name
            try:
                attr_obj = getattr(module, attr_name)
                if callable(attr_obj):
                    if full_name not in func_list:
                        func_list.append(full_name)
                        doc = get_doc(attr_obj, full_name)
                        if doc: api_doc_dict[full_name] = doc
            except: ...
            try:
                crawl_callable_attributes(attr_obj, library_name)
            except: ...
    else:   # if the module is a package
        # use pkgutil to traverse all the files and actively import them to deal with lazy import
        for module_info in pkgutil.iter_modules(path=module.__path__):
            file_name = module_info.name
            # filter some default and internal modules
            # if file_name == '_xla_ops': continue
            # if file_name in ['setup', 'tests'] or file_name.startswith('_'): continue
            # if file_name in ['setup', 'tests', '_testing', '_libs']: continue
            if file_name.startswith('_'): continue
            # actively import the module
            try:
                full_name = module_name + '.' + file_name
                submodule = importlib.import_module(full_name)
            except:
                print(full_name)
                continue
            crawl_callable_attributes(submodule, library_name)


def crawl_python_doc(library_list):
    for lib_name in library_list:
        lib = importlib.import_module(lib_name)
        crawl_callable_attributes(lib, lib.__name__)


if __name__ == '__main__':
    library_list = third_party_lib_list
    api_sign_file = '../data/python_docs/api_sign_third_party.txt'
    api_doc_file = '../data/python_docs/api_doc_third_party.json'
    # library_list = py_builtin_lib_list
    # api_sign_file = '../data/python_docs/api_sign_builtin.txt'
    # api_doc_file = '../data/python_docs/api_doc_builtin.json'

    func_list = list()  # store each function's full name
    module_list = list()    # store each module
    api_doc_dict = dict()   # store api docs in a dict, key is full name, value is docs

    crawl_python_doc(library_list)

    with open(api_sign_file, 'w+') as f:
        for func in func_list:
            f.write(str(func) + '\n')
    with open(api_doc_file, 'w+') as f:
        json.dump(api_doc_dict, f, indent=2)
