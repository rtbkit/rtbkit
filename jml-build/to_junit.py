#make test | python junit.py > testresults.xml

import os, fileinput
from xml.sax.saxutils import escape

passed = set()
failed = set()

for l in fileinput.input():
    pieces = l.strip().replace("\x1b[0m", "").replace("\x1b[32m","").replace("\x1b[31m", "").split()
    if not len(pieces) == 2: continue
    if pieces[1] == "passed": passed.add(pieces[0])
    if pieces[1] == "FAILED": failed.add(pieces[0])

failed.difference_update(passed)

print """<?xml version="1.0" encoding="UTF-8" ?>
<testsuite errors="0" tests="%d" time="0" failures="%d" name="tests">""" % (len(passed)+len(failed), len(failed))

for f in failed:

    failContent = ""

    if os.path.isfile("build/x86_64/tests/%s.failed" % f):
	with open("build/x86_64/tests/%s.failed" % f, "r") as failFile:
            failContent = failFile.read()

    print """    <testcase time="0" name="%s">
        <failure type="failure" message="Check log">%s</failure>
    </testcase>""" % (f, escape(failContent))
for p in passed: 
    print """    <testcase time="0" name="%s"/>""" % p

print "</testsuite>"
