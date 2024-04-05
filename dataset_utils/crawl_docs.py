import inspect
import os, sys

# common used data science libs
import tensorflow
import matplotlib
import sklearn
import numpy
import torch
import pandas
import scipy

third_party_lib_list = [tensorflow, matplotlib, sklearn, numpy, torch, pandas, scipy]

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

def crawl_callable_attributes(module, library_name):
    if not module.__name__.startswith(library_name): return     # filter module not belong to this lib
    if module in module_list: return    # filter already traversed module
    module_list.append(module)

    for attr_name in dir(module):
        try:
            attr_obj = getattr(module, attr_name)
        except:
            # print(module.__name__ + '.' + attr_name)
            continue
        # attr_obj = getattr(module, attr_name)
        if callable(attr_obj):
            if attr_name.startswith('_'): continue
            full_name = module.__name__ + '.' + attr_name
            if full_name not in func_list: func_list.append(full_name)
        elif inspect.ismodule(attr_obj):
            crawl_callable_attributes(attr_obj, library_name)

def crawl_api_signs(library_list):
    for library in library_list:
        crawl_callable_attributes(library, library.__name__)

def get_doc(api_sign):
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        help(api_sign)
        doc = buffer.getvalue()
        helped_api_sign, content = doc.split('\n\n',1)
        # use the same way in match oracle docs to identify api sign and verify if its same
        if 'built-in' in helped_api_sign:
            module = 'builtins'
        else:
            module = helped_api_sign.split('module ')[1].replace(':', '')
        method = api_sign.rsplit('.', 1)[1]
        helped_api_sign = module + '.' + method
        assert helped_api_sign == api_sign
    except:
        sys.stdout = sys.__stdout__
        print(api_sign)
        content = None
    finally:
        sys.stdout = sys.__stdout__

    return content


def crawl_api_docs(api_sign_list):
    api_doc_dict = dict()
    for api_sign in api_sign_list:
        content = get_doc(api_sign)
        if content is not None:
            api_doc_dict[api_sign] = content

    return api_doc_dict


if __name__ == '__main__':
    # library_list = third_party_lib_list
    # api_sign_file = '../data/python_docs/api_sign_third_party.txt'
    # api_doc_file = '../data/python_docs/api_doc_third_party.json'
    library_list = py_builtin_lib_list
    api_sign_file = '../data/python_docs/api_sign_builtin.txt'
    api_doc_file = '../data/python_docs/api_doc_builtin.json'

    # crawl api sign
    func_list = list()
    module_list = list()
    crawl_api_signs(library_list)
    with open(api_sign_file, 'w+') as f:
        for func in func_list:
            f.write(str(func) + '\n')

    # crawl api docs based on signs
    with open(api_sign_file, 'r') as f:
        api_sign_list = [line.strip() for line in f.readlines()]
    api_doc_dict = crawl_api_docs(api_sign_list)
    with open(api_doc_file, 'w+') as f:
        json.dump(api_doc_dict, f, indent=2)