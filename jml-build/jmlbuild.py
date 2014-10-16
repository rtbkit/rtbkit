#! /usr/bin/python
#------------------------------------------------------------------------------#
# builddeps.py
# by Remi Attab on 29-10-2012
#
# Parses the given jml-build Makefile and returns a graph of all (well most)
# build dependencies for the supported target types.
#
# Supported target types are listed within Parser.func_map and is fairly simple
# to extend.
#
# The public interface of this module consists of Ext, Graph and
# parse_makefile(). The verbose, strict and quiet flags can also be used to
# change the chattyness of the script on stdout. Nothing else should be
# accessed.
#
#------------------------------------------------------------------------------#


import os
import re
import sys
import traceback


#------------------------------------------------------------------------------#
# DEBUG                                                                        #
#------------------------------------------------------------------------------#

verbose = False
strict = False
quiet = True

def print_dbg(msg):
    global verbose
    if verbose: print "// " + msg

def print_info(msg):
    global quiet
    if not quiet: print "// " + msg

def print_err(err, msg):
    global quiet
    if not quiet:
        errstr = "(%s)" % err if err else ""
        print "// ERROR%s: %s" % (errstr, msg)


#------------------------------------------------------------------------------#
# GRAPH                                                                        #
#------------------------------------------------------------------------------#

class Ext:
    SO            = ".so"
    EXE           = ".exe"
    TEST          = ".test"
    MK            = ".mk"
    NODEJS_ADDON  = ".nodejs_addon"
    NODEJS_MODULE = ".nodejs_module"
    NODEJS_TEST   = ".nodejs_test"

class Graph:
    def __init__(self):
        self.edges = {}
        self.order = []

    def add_vertex(self, src):
        if src in self.edges: return

        print_dbg("%s -> None" % src)
        self.edges[src] = []
        self.order.append(src)

    def add_edge(self, src, dest):
        print_dbg("%s -> %s" % (src, dest))

        self.add_vertex(src)
        self.edges[src].append(dest)


#------------------------------------------------------------------------------#
# LEXER                                                                        #
#------------------------------------------------------------------------------#

class Token:
    START_BLOCK = 10
    END_BLOCK   = 11

    EQUAL       = 22
    FILE_SEP    = 23
    PLUS_EQUAL  = 24
    COND_EQUAL  = 25

    WORD        = 31
    ARG_SEP     = 32


def strip_line(line):
    return line.strip(" \t\n")


def next_token_impl(line):
    """
    Very simple lexer for our parser that returns the <token, updated line>
    tuple. The token is a list composed of a member of the Token class and
    followed by a series of values associated with the token.
    """

    dbg = "\t\ttoken: %s -> " % line

    if line[0] == ",":
        print_dbg(dbg + "SEP")
        return [Token.ARG_SEP], line[1:]

    if line[0:2] == "$(":
        print_dbg(dbg + "START")
        return [Token.START_BLOCK], line[2:]

    if line[0] == ")":
        print_dbg(dbg + "END")
        return [Token.END_BLOCK], line[1:]

    if line[0:2] == ":=":
        print_dbg(dbg + "EQUAL")
        return [Token.EQUAL], line[2:]

    if line[0:2] == "+=":
        print_dbg(dbg + "PLUS_EQUAL")
        return [Token.PLUS_EQUAL], line[2:]

    if line[0:2] == "?=":
        print_dbg(dbg + "COND_EQUAL")
        return [Token.COND_EQUAL], line[2:]

    m = re.match("([\w_\./\-+]+)", line)
    if m:
        word = m.group(1)
        print_dbg(dbg + "WORD(%s)" % word)
        return [Token.WORD, word], line[len(word):]

    return [None], ""


def next_token(line):
    """
    Simple wrapper for our lexer that ensures that all returned strings are
    properly stripped of white spaces.
    """
    tok, line = next_token_impl(line)
    return tok, strip_line(line)


def peek_token(line):
    """
    Returns the next token without removing it from line
    """
    tok, line = next_token_impl(line)
    return tok


def token_value(tok):
    return tok[1]

def accept(tok, expected):
    return tok[0] == expected

def expect(tok, expected):
    assert accept(tok, expected)


def next_line(stream):
    """
    Returns the next line that needs parsing. The returned line is guaranteed to
    be striped of leading and trailing white spaces. We also ensure that a line
    broken up by the \ character will be returned as a single line.
    """
    result = ""
    while True:
        line = stream.readline()
        if len(line) == 0: break

        line = strip_line(line)

        # Continue on empty line only if we don't have anything. This is a
        # workaround for having a trailing slash at the end of a variable
        # definition.
        if len(line) == 0:
            if len(result) == 0: continue
            else: break

        if len(result) > 0:
            result += " "

        # read the next line if we have a trailing slash.
        if line[-1] == "\\":
            result += strip_line(line[:-1])
            continue

        result += line
        break

    if len(result) == 0:
        print_dbg("\tEOF")
        return None

    print_dbg("\tline: " + result)
    return result



#------------------------------------------------------------------------------#
# PARSER                                                                       #
#------------------------------------------------------------------------------#

class Parser:

    def __init__(self):

        # Contains the current file being parsed. Used to track the mk files
        # dependencies.
        self.current_file = ""

        # Maps function names to its parser
        self.func_map = {
            "include_sub_makes" : self.parse_func_sub_make,
            "include_sub_make"  : self.parse_func_sub_make,
            "library"           : self.parse_func_library,
            "program"           : self.parse_func_program,
            "test"              : self.parse_func_test,
            "nodejs_addon"      : self.parse_func_nodejs_addon,
            "nodejs_module"     : self.parse_func_nodejs_module,
            "nodejs_program"    : self.parse_func_nodejs_module,
            "vowscoffee_test"   : self.parse_func_nodejs_test,
            "nodejs_test"       : self.parse_func_nodejs_test
            }

        # Variable map used for macro expansion
        self.var_map = {}

        # Keeps track of the current folder.
        self.folder_stack = []

        # Keeps track of where we've been.
        self.visited_files = set([])

        # make keywords that can show up at the begining of a line
        self.keywords = ['include', '-include', 'export', 'ifeq', 'endif']

        # Dependency graph
        self.graph = Graph()


    def include(self, folder, makefile):
        self.graph.add_edge(self.current_file, makefile)
        self.parse_makefile(folder, makefile)


    # FUNCTION PARSER
    #--------------------------------------------------------------------------#
    # Parsers defined here have to be registerd in the func_map map so that the
    # generic parser can find it.

    def parse_func_sub_make(self, line):
        """
        Parses the include_submake function and recursively parses the makefiles
        it points to.
        """
        print_dbg("\tsub_make: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        if len(params) == 1:
            for makefile in params[0]:
                self.include(makefile, makefile + Ext.MK)

        elif len(params) == 2:
            assert len(params[0]) == 1
            assert len(params[1]) == 1
            self.include(params[1][0], params[0][0] + Ext.MK)

        else:
            assert len(params[2]) == 1
            filename = params[2][0]
            folder = params[1][0] if len(params[1]) > 0 else params[0][0]
            self.include(folder, filename)

        return line


    def parse_func_library(self, line):
        """
        Parses the library function params and adds the relevant dependencies.
        """
        print_dbg("\tlibrary: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) >= 1
        assert len(params[0]) == 1

        libname = params[0][0] + Ext.SO
        self.graph.add_edge(self.current_file, libname)
        self.graph.add_vertex(libname)

        deps = params[2] if len(params) >= 3 else []
        for dep in deps:
            self.graph.add_edge(libname, dep + Ext.SO)

        return line


    def parse_func_program(self, line):
        """
        Parser for the program target params.
        """
        print_dbg("\tprogram: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        assert len(params[0]) == 1
        program = params[0][0] + Ext.EXE
        self.graph.add_edge(self.current_file, program)
        self.graph.add_vertex(program)

        if len(params) > 1:
            for lib in params[1]:
                self.graph.add_edge(program, lib + Ext.SO)

        return line


    def parse_func_test(self, line):
        """
        Parser for the test target params.
        """
        print_dbg("\ttest: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        assert len(params[0]) == 1
        test = params[0][0] + Ext.TEST
        self.graph.add_edge(self.current_file, test)
        self.graph.add_vertex(test)

        if len(params) > 1:
            for lib in params[1]:
                self.graph.add_edge(test, lib + Ext.SO)

        return line


    def parse_func_nodejs_addon(self, line):
        """
        Parses for the nodejs addon params and adds the relevant dependencies
        """
        print_dbg("\tnodejs_addon: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        assert len(params[0]) == 1
        addon = params[0][0] + Ext.NODEJS_ADDON
        self.graph.add_edge(self.current_file, addon)
        self.graph.add_vertex(addon)

        if len(params) > 1:
            sources = params[1]

        if len(params) > 2:
            for lib in params[2]:
                self.graph.add_edge(addon, lib + Ext.SO)

        if len(params) > 3:
            for libjs in params[3]:
                self.graph.add_edge(addon, libjs + Ext.NODEJS_ADDON)

        return line


    def parse_func_nodejs_module(self, line):
        """
        Parses for the nodejs module params and adds the relevant dependencies
        """
        print_dbg("\tnodejs_module: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        assert len(params[0]) == 1
        module = params[0][0] + Ext.NODEJS_MODULE
        self.graph.add_edge(self.current_file, module)
        self.graph.add_vertex(module)

        if len(params) > 1:
            assert len(params[1]) == 1
            sources = params[1][0]

        # Both modules and addon can be specified in the same one param. A good
        # educated guess is that our dependency is built before our library. And
        # by good I mean laughable notion that a build system would retain some
        # kind of sane structure...
        if len(params) > 2:
            for lib in params[2]:
                if lib + Ext.NODEJS_ADDON in self.graph.edges:
                    self.graph.add_edge(module, lib + Ext.NODEJS_ADDON)
                else:
                    self.graph.add_edge(module, lib + Ext.NODEJS_MODULE)

        return line


    def parse_func_nodejs_test(self, line):
        """
        Parses for the nodejs and vows test params and adds the relevant
        dependencies.
        """
        print_dbg("\tvowscoffee_test: " + line)

        params, line = self.parse_func_params(line)
        assert len(params) > 0

        assert len(params[0]) == 1
        module = params[0][0] + Ext.NODEJS_TEST
        self.graph.add_edge(self.current_file, module)
        self.graph.add_vertex(module)

        # Both modules and addon can be specified in the same one param. A good
        # educated guess is that our dependency is built before our library. And
        # by good I mean laughable notion that a build system would retain some
        # kind of sane structure...
        if len(params) > 1:
            for lib in params[1]:
                if lib + Ext.NODEJS_ADDON in self.graph.edges:
                    self.graph.add_edge(module, lib + Ext.NODEJS_ADDON)
                else:
                    self.graph.add_edge(module, lib + Ext.NODEJS_MODULE)

        return line


    # GENERIC PARSER
    #--------------------------------------------------------------------------#

    def parse_var_decl(self, var, line):
        """
        Parse the declaration of a variable and add it to the var_map.
        """
        tok, line = next_token(line)

        if accept(tok, Token.EQUAL):
            self.var_map[var] = line
            print_dbg("\tvar: %s = %s" %(var, self.var_map[var]))
            return ""

        elif accept(tok, Token.PLUS_EQUAL):
            if var in self.var_map:
                self.var_map[var] += line
            else:
                self.var_map[var] = line

            print_dbg("\tvar: %s = %s" %(var, self.var_map[var]))
            return ""

        elif accept(tok, Token.COND_EQUAL):
            if var not in self.var_map:
                self.var_map[var] = line
            print_dbg("\tvar: %s = %s" %(var, self.var_map[var]))
            return ""

        assert False


    def parse_arg_list(self, line):
        """
        Parse a list of arguments seperated by spaces.
        """
        tok = peek_token(line)
        args = []

        print_dbg("\targ_list")

        while tok:
            if accept(tok, Token.WORD):
                tok, line = next_token(line)
                args.append(token_value(tok))

            elif accept(tok, Token.START_BLOCK):
                tok, line = next_token(line)

                tok, line = next_token(line)
                var = token_value(tok)

                tok, line = next_token(line)
                expect(tok, Token.END_BLOCK)

                if var in self.var_map:
                    line = self.var_map[var] + " " + line
                    print_dbg("\tmap: %s -> %s" % (var, self.var_map[var]))
                else:
                    print_dbg("unknown: " + var)

            else:
                break

            tok = peek_token(line)

        return args, line


    def parse_func_params(self, line):
        """
        Parses a list of function parameters sperated by commas.
        """
        params = []

        while True:
            # If the next token isn't a comma then stop.
            if not accept(peek_token(line), Token.ARG_SEP):
                break
            tok, line = next_token(line)

            expect(tok, Token.ARG_SEP)
            args, line = self.parse_arg_list(line)

            params.append(args)

        return params, line


    def parse_func_default(self, line):
        """
        Function which don't have special handlers are parsed here.
        """
        print_dbg("\tdefault_func: " + line)
        params, line = self.parse_func_params(line)
        return line


    def parse_function(self, line):
        """
        Parses the common parts of the function and dispatches the params
        parsing to a sub parser depending on the function name.
        """
        print_dbg("\tfunction: " + line)

        tok, line = next_token(line)
        expect(tok, Token.START_BLOCK)

        tok, line = next_token(line)
        expect(tok, Token.WORD)
        assert token_value(tok) == "call"

        tok, line = next_token(line)
        expect(tok, Token.WORD)

        # Dispatch to the appropriate function parser
        func = token_value(tok)
        if func in self.func_map:
            line = self.func_map[func](line)
        else:
            line = self.parse_func_default(line)

        tok, line = next_token(line)
        expect(tok, Token.END_BLOCK)

        return line

    def parse_include(self, line):
        m = re.match("([\w_]+\.mk)", line)
        if not m: return line

        self.include('.', line)
        return ''

    def parse_line(self, line, path):
        try:
            tok, line = next_token(line)

            if accept(tok, Token.WORD):
                word = token_value(tok)

                if word == "include" or word == "-include":
                    line = self.parse_include(line)

                if word not in self.keywords:
                    line = self.parse_var_decl(word, line)

            elif accept(tok, Token.START_BLOCK):
                tok, line = next_token(line)
                expect(tok, Token.WORD)

                # We don't bother with any non-eval function calls.
                if token_value(tok) == "eval":
                    line = self.parse_function(line)
                    tok, line = next_token(line)
                    expect(tok, Token.END_BLOCK)

        except Exception as ex:
            print_err(str(ex), "%s: %s" % (path, line))
            if verbose: print traceback.format_exc()
            if strict: raise ex


    def parse_makefile(self, folder, filename):
        """
        Iterates over a file parsingall the var declaration and eval function
        vals.
        """


        self.folder_stack.append(folder)

        path = '/'.join(self.folder_stack) + "/" + filename

        if path in self.visited_files:
            print_dbg("been-there-done-that: " + path)
            self.folder_stack.pop()
            self.current_file = old_file
            return

        self.visited_files = self.visited_files | set([path])
        print_info("FILE: " + path)

        old_file = self.current_file
        self.current_file = filename

        try:
            line = ""
            with open(path, 'r') as f:
                line = next_line(f)
                while line:
                    self.parse_line(line, path)
                    line = next_line(f)

        except Exception as ex:
            print_err(str(ex), "%s: %s" % (path, line))
            if verbose: print traceback.format_exc()
            if strict: raise ex

        self.folder_stack.pop()
        self.current_file = old_file


#------------------------------------------------------------------------------#
# PUBLIC INTERFACE                                                             #
#------------------------------------------------------------------------------#

def parse_makefile(makefile = "Makefile", folder = "."):
    """
    Parses a given makefile and returns a graph object which declares the
    dependencies.
    """

    parser = Parser()
    parser.parse_makefile(folder, makefile)
    return parser.graph


def find_dotgit(folder):
    """
    Finds the parent folder that has a .git folder. Since this script will
    usually be invoked from within a submodule, the Makefile in the submodule
    will can't be parsed properly. So we need to find the real one which is
    located at the root of our git repo.
    """
    if os.path.isdir(os.path.join(folder, ".git")):
        return folder
    return find_dotgit(os.path.dirname(folder))

