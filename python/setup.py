#!/usr/bin/python
# -*- coding: utf-8 -*-
#  Copyright (C) 2001-2003, 2008-2009 Francis Piéraut <fpieraut@gmail.com>
#  Copyright (C) 2009 Yannick Gingras <ygingras@ygingras.net>

#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.

#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

''' Installer for the `jml` python API wrapper.  To install, run:  
  python setup.py build_ext --inplace or python setup.py install'''

# from distutils.core import setup, Extension
from setuptools import setup, Extension
from setuptools import find_packages
from subprocess import call
import os

import sys
sys.path.append("jml")

from metainfo import __version__, JML_CORE_FILES

jml_modules_files = [os.path.join("jml", "src", f) 
                         for f in JML_CORE_FILES]
# To create jml_wrap.cxx you have to call swig like that: swig -python -c++ jml.i
for filename in ['python/feature']:
    print "creating %s_wrap.cxx" % filename
    call(['swig', '-python', '-c++', '%s.i' % filename])
    jml_modules_files.extend(['%s_wrap.cxx' % filename]) 

jml_extension = Extension("_jml", sources=jml_modules_files)

setup(name='jml',
      version=__version__,
      author='Jeremy Barnes',
      author_email='jeremy@barneso.com',
      description = "python jml wrapper interface",
      license="AGPL v3 or later",
      url='http://hello.world/',
      packages = find_packages(),
      ext_modules = [jml_extension],
      include_package_data=True,
      #setup_requires = ['swig>1.3.35']
)
