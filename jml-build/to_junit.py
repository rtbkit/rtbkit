#
# to_junit.py
# Nicolas Kructhen, 2013-03-26
# Mich, 2015-03-05
# Copyright (c) 2013 Datacratic Inc.  All rights reserved.
#
# make test | python junit.py > testresults.xml

import os
import fileinput
import re
from xml.etree import ElementTree
from StringIO import StringIO

# thanks to
# https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
ansi_escape = re.compile(r'\x1b[^m]*m')


passed = set()
failed = set()

for l in fileinput.input():
#    pieces = l.strip().replace("\x1b[0m", "").replace("\x1b[32m", "") \
#        .replace("\x1b[31m", "").split()
    pieces = l.strip().replace(chr(27), "").split()
    if not len(pieces) == 2:
        continue
    if pieces[1] == "passed":
        passed.add(pieces[0])
    if pieces[1] == "FAILED":
        failed.add(pieces[0])

failed.difference_update(passed)

builder = ElementTree.TreeBuilder()
builder.start('testsuite', {
    'errors'   : '0',
    'tests'    : str(len(passed) + len(failed)),
    'time'     : '0',
    'failures' : str(len(failed)),
    'name'     : 'tests'
})

for f in failed:
    fail_content = ""

    if os.path.isfile("build/x86_64/tests/%s.failed" % f):
        with open("build/x86_64/tests/%s.failed" % f, "r") as failFile:
            fail_content = failFile.read().replace(chr(27), "")
    builder.start('testcase', {
        'time' : '0',
        'name' : f
    })
    builder.start('failure', {
        'type' : 'failure',
        'message' : 'Check log'
    })
    ansi_escape.sub('', fail_content)
    builder.data(unicode(fail_content, 'utf-8'))
    builder.end('failure')
    builder.end('testcase')

for p in passed:
    builder.start('testcase', {
        'time' : '0',
        'name' : p
    })
    builder.end('testcase')

builder.end('testsuite')

tree = ElementTree.ElementTree()
element = builder.close()
tree._setroot(element)
io = StringIO()
tree.write(io, encoding='utf-8', xml_declaration=True)
print io.getvalue()
