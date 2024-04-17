import inspect, pkgutil, importlib
import io, os, sys, json

# common used data science libs
import tensorflow
import matplotlib
import sklearn
import numpy
import torch
import pandas
import scipy
import seaborn

third_party_lib_list = [tensorflow, matplotlib, sklearn, numpy, torch, pandas, scipy, seaborn]


# python default libs from https://docs.python.org/3.7/library/index.html
import builtins
import string, re, difflib, textwrap, unicodedata, stringprep, readline, rlcompleter
import struct, codecs
import datetime, calendar, collections, collections.abc, heapq, bisect, array, weakref, types, copy, pprint, reprlib, enum
import numbers, math, cmath, decimal, fractions, random, statistics
import itertools, functools, operator
import pathlib, os.path, fileinput, stat, filecmp, tempfile, glob, fnmatch, linecache, shutil, macpath
import pickle, copyreg, shelve, marshal, dbm, sqlite3
import zlib, gzip, bz2, lzma, zipfile, tarfile
import csv, configparser, netrc, xdrlib, plistlib
import hashlib, hmac, secrets
import os, io, time, argparse, getopt, logging, logging.config, logging.handlers, getpass, curses, curses.textpad, curses.ascii, curses.panel, platform, errno, ctypes
import threading, multiprocessing, concurrent.futures, subprocess, sched, queue, _thread, _dummy_thread, dummy_threading
import contextvars, asyncio
import asyncio, socket, ssl, select, selectors, asyncore, asynchat, signal, mmap
import email, json, mailcap, mailbox, mimetypes, base64, binhex, binascii, quopri, uu
import html, html.parser, html.entities, xml
import webbrowser, cgi, cgitb, wsgiref, urllib, urllib.request, urllib.response, urllib.parse, urllib.error, urllib.robotparser, http, http.client, ftplib, poplib, imaplib, nntplib, smtplib, smtpd, telnetlib, uuid, socketserver, http.server, http.cookies, http.cookiejar, xmlrpc, xmlrpc.client, xmlrpc.server, ipaddress
import audioop, aifc, sunau, wave, chunk, colorsys, imghdr, sndhdr, ossaudiodev
import gettext, locale
import turtle, cmd, shlex
import tkinter, tkinter.ttk, tkinter.tix, tkinter.scrolledtext
import typing, pydoc, doctest, unittest, unittest.mock, lib2to3, test, test.support, test.support.script_helper
import bdb, faulthandler, pdb, cProfile, profile, timeit, trace, tracemalloc
import distutils, ensurepip, venv, zipapp
import sys, sysconfig, builtins, __main__, warnings, dataclasses, contextlib, abc, atexit, traceback, __future__, gc, inspect, site
import code, codeop
import zipimport, pkgutil, modulefinder, runpy, importlib
import parser, ast, symtable, symbol, token, keyword, tokenize, tabnanny, pyclbr, py_compile, compileall, dis, pickletools
# import formatter
# import mislib, msvcrt, winreg, winsound
# import posix, pwd, spwd, grp, crypt, termios, tty, pty, fcntl, pipes, resource, nis, syslog
# import optparse, imp

py_builtin_lib_list = [
    builtins,
    string, re, difflib, textwrap, unicodedata, stringprep, readline, rlcompleter,
    struct, codecs,
    datetime, calendar, collections, collections.abc, heapq, bisect, array, weakref, types, copy, pprint, reprlib, enum,
    numbers, math, cmath, decimal, fractions, random, statistics,
    itertools, functools, operator,
    pathlib, os.path, fileinput, stat, filecmp, tempfile, glob, fnmatch, linecache, shutil, macpath,
    pickle, copyreg, shelve, marshal, dbm, sqlite3,
    zlib, gzip, bz2, lzma, zipfile, tarfile,
    csv, configparser, netrc, xdrlib, plistlib,
    hashlib, hmac, secrets,
    os, io, time, argparse, getopt, logging, logging.config, logging.handlers, getpass, curses, curses.textpad, curses.ascii, curses.panel, platform, errno, ctypes,
    threading, multiprocessing, concurrent.futures, subprocess, sched, queue, _thread, _dummy_thread, dummy_threading,
    contextvars, asyncio,
    asyncio, socket, ssl, select, selectors, asyncore, asynchat, signal, mmap,
    email, json, mailcap, mailbox, mimetypes, base64, binhex, binascii, quopri, uu,
    html, html.parser, html.entities, xml,
    webbrowser, cgi, cgitb, wsgiref, urllib, urllib.request, urllib.response, urllib.parse, urllib.error, urllib.robotparser, http, http.client, ftplib, poplib, imaplib, nntplib, smtplib, smtpd, telnetlib, uuid, socketserver, http.server, http.cookies, http.cookiejar, xmlrpc, xmlrpc.client, xmlrpc.server, ipaddress,
    audioop, aifc, sunau, wave, chunk, colorsys, imghdr, sndhdr, ossaudiodev,
    gettext, locale,
    turtle, cmd, shlex,
    tkinter, tkinter.ttk, tkinter.tix, tkinter.scrolledtext,
    typing, pydoc, doctest, unittest, unittest.mock, lib2to3, test, test.support, test.support.script_helper,
    bdb, faulthandler, pdb, cProfile, profile, timeit, trace, tracemalloc,
    distutils, ensurepip, venv, zipapp,
    sys, sysconfig, builtins, __main__, warnings, dataclasses, contextlib, abc, atexit, traceback, __future__, gc, inspect, site,
    code, codeop,
    zipimport, pkgutil, modulefinder, runpy, importlib,
    parser, ast, symtable, symbol, token, keyword, tokenize, tabnanny, pyclbr, py_compile, compileall, dis, pickletools,
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


# def crawl_callable_attributes(module, library_name):
#     if not module.__name__.startswith(library_name): return     # filter module not belong to this lib
#     if module in module_list: return    # filter already traversed module
#     module_list.append(module)
#
#     for attr_name in dir(module):
#         try:
#             attr_obj = getattr(module, attr_name)
#         except:
#             # print(module.__name__ + '.' + attr_name)
#             continue
#         # attr_obj = getattr(module, attr_name)
#         if callable(attr_obj):
#             if attr_name.startswith('_'): continue
#             full_name = module.__name__ + '.' + attr_name
#             if full_name not in func_list:
#                 func_list.append(full_name)
#                 # content = get_doc(attr_obj, full_name)
#                 # if content is not None: api_doc_dict[full_name] = content
#         elif inspect.ismodule(attr_obj):
#             crawl_callable_attributes(attr_obj, library_name)


def crawl_callable_attributes(module, library_name):
    global item_count
    if not module.__name__.startswith(library_name): return  # filter module not belong to this lib
    if module in module_list: return    # filter already traversed module
    module_list.append(module)

    if not hasattr(module, '__path__'):
        for attr_name in dir(module):
            full_name = module.__name__ + '.' + attr_name
            try:
                attr_obj = getattr(module, attr_name)
                if callable(attr_obj):
                    if attr_name.startswith('_'): continue
                    if full_name not in func_list:
                        func_list.append(full_name)
                        doc = get_doc(attr_obj, full_name)
                        if doc: api_doc_dict[full_name] = doc
            except:
                # print(full_name)
                continue

    else:
        for module_info in pkgutil.iter_modules(path=module.__path__):
            file_name = module_info.name
            # filter some default and internal modules
            # if file_name == '_xla_ops': continue
            if file_name in ['setup', 'tests'] or file_name.startswith('_'): continue
            full_name = module.__name__ + '.' + file_name
            # each submodule is a py file or a dir, import the submodule to deal with lazy import
            try:
                submodule = importlib.import_module(full_name)
            except:
                # print(full_name)
                continue
            crawl_callable_attributes(submodule, library_name)



def crawl_python_doc(library_list):
    for library in library_list:
        crawl_callable_attributes(library, library.__name__)


if __name__ == '__main__':
    library_list = third_party_lib_list
    api_sign_file = '../data/python_docs/api_sign_third_party_new.txt'
    api_doc_file = '../data/python_docs/api_doc_third_party_new.json'
    # library_list = py_builtin_lib_list
    # api_sign_file = '../data/python_docs/api_sign_builtin_new.txt'
    # api_doc_file = '../data/python_docs/api_doc_builtin_new.json'

    func_list = list()
    module_list = list()
    api_doc_dict = dict()
    item_count = 0


    crawl_python_doc(library_list)

    with open(api_sign_file, 'w+') as f:
        for func in func_list:
            f.write(str(func) + '\n')
    with open(api_doc_file, 'w+') as f:
        json.dump(api_doc_dict, f, indent=2)

    # library_list = third_party_lib_list
    # api_sign_file = '../data/python_docs/api_sign_third_party_new.txt'
    # api_doc_file = '../data/python_docs/api_doc_third_party_new.json'
    #
    # import json
    #
    # with open(api_sign_file, 'r') as f:
    #     existing_func_list = [line.strip() for line in f.readlines()]
    # with open(api_doc_file, 'r') as f:
    #     existing_api_doc_dict = json.load(f)
    # assert len(existing_func_list) == len(existing_api_doc_dict.items())
    #
    # import seaborn
    #
    # func_list = []
    # module_list = []
    # api_doc_dict = dict()
    # crawl_callable_attributes(seaborn, 'seaborn')
    #
    # for full_name in func_list:
    #     assert full_name not in existing_func_list
    #     existing_func_list.append(full_name)
    #     existing_api_doc_dict[full_name] = api_doc_dict[full_name]
    #
    # with open(api_sign_file, 'w+') as f:
    #     for func in existing_func_list:
    #         f.write(str(func) + '\n')
    # with open(api_doc_file, 'w+') as f:
    #     json.dump(existing_api_doc_dict, f, indent=2)
