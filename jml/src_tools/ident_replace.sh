#!/bin/bash
# ident_replace.sh
# Jeremy Barnes, 20 October 2000
# Copyright (c) 2000 Idilia Inc.  All rights reserved.
# $Id$

# This is a script that replaces all occurances of an identifier within the
# current tree to another identifier.  It creates backup files.

# Usage ident_replace.sh <identifier> <replacement>

STRING=$1
REPLACEMENT=$2
FILE=$3

#echo "Replacing $STRING with $REPLACEMENT..."

#$FILES=$(grep -r -i -I -l "$STRING" * | grep -v '~' | grep -v '#')

#echo $FILES

#exit

#for FILE in $FILES; do
    echo $FILE
    mv -f $FILE "$FILE~";
    cat "$FILE~" | sed "s/$STRING/$REPLACEMENT/g" > $FILE;
    chmod --reference=$FILE~ $FILE
#done;