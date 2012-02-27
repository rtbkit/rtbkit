#!/bin/bash --norc

if [ -d /usr/include/python2.7 ] ; then
    echo "2.7";
    exit 0;
fi

if [ -d /usr/include/python2.6 ] ; then
    echo "2.6";
    exit 0;
fi

echo "unknown";
exit 0
