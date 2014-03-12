#!/bin/sh
# Script to search for the given thing throughout the source code
# Jeremy Barnes, 20 August 2003
# Copyright (c) 2003 Idilia Inc.  All rights reserved.
# $Source$
# $Id$

if [ $# -eq 0 ]; then
    echo "Tool to search through source code for the given grep regex."
    echo
    echo "usage: $0 [options] expression [dir]"
    echo
    echo "options:"
    echo "   -i     perform case-insensitive search"
    echo 
    echo "The directory will be inferred automatically if not given."
    exit
fi

while getopts i option
  do
  case $option in
      i) GREPOPT=-i;;
      *) echo option $option
  esac
done

#echo optind expr $OPTIND

shift `expr $OPTIND - 1`
WHAT=$1
DIR=$2

#echo '$1' $1 '$2' $2
#echo "DIR=$DIR"

#echo "dir = '$DIR'"
#echo "-d DIR = "
#test -n $DIR
#echo $?

#if [ -d $DIR ]; then
#    echo "dir does exist"
#else
#    echo "dir doesn't exist"
#fi

if [ -z $DIR ]
    then
    #echo "in fixup dir"
    if [ -d $0/../utilities/buffer.h ]; then
        DIR=$0/..
    elif [ -f utilities/buffer.h ]; then
        DIR="."
    elif [ -d ../src ]; then
        DIR="../src"
    elif [ -d src ]; then
        DIR="src"
    else
        DIR="."
    fi
fi

#echo "dir = " $DIR
#echo "what = " $WHAT
#echo "grepopt = " $GREPOPT

find $DIR -name "*.cc" -print0 -or -name "*.h" -print0 -or -name "*.f" -print0 \
    | xargs -0 grep $GREPOPT "$WHAT"
