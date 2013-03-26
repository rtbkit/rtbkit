#make test | python junit.py > testresults.xml

import fileinput

passed = set()
failed = set()

for l in fileinput.input():
    pieces = l.strip().replace("\x1b[0m", "").replace("\x1b[32m","").replace("\x1b[31m", "").split()
    if not len(pieces) == 2: continue
    if pieces[1] == "passed": passed.add(pieces[0])
    if pieces[1] == "FAILED": failed.add(pieces[0])

print """<?xml version="1.0" encoding="UTF-8" ?>
<testsuite errors="0" tests="%d" time="0" failures="%d" name="tests">""" % (len(passed)+len(failed), len(failed))

for f in failed: 
    print """    <testcase time="0" name="%s">
        <failure type="failure" message="Check log" />
    </testcase>""" % f
for p in passed: 
    print """    <testcase time="0" name="%s"/>""" % p

print "</testsuite>"
