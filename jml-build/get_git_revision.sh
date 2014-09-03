#!/bin/bash

# Script to get the git revision ID for the current repository

# Arguments

# 1.  Make target

target=$1

#set -v
#set -j
set -e

DIRTY=0

ESC=''

COLOR_RED=$ESC[31m
COLOR_GREEN=$ESC[32m
COLOR_YELLOW=$ESC[33m
COLOR_BLUE=$ESC[34m
COLOR_VIOLET=$ESC[35m
COLOR_CYAN=$ESC[36m
COLOR_RESET=$ESC[0m
COLOR_WHITE=$ESC[37m
COLOR_BOLD=$ESC[1m

COLOR_ERROR=$COLOR_BOLD$COLOR_YELLOW

echo -n "Checking number of non-clean files and submodules according to git... " > /dev/stderr
NUMUNCLEAN=`git status --porcelain | grep '^ M\|^??' | wc -l`
echo $NUMUNCLEAN > /dev/stderr

# The rest we only do if the directory isn't clean.  If it's clean then it will always come
# up negative
if [ $NUMUNCLEAN -gt 0 ]; then

    mkdir -p $BUILD/tmp
    UNADDEDFILES=`pwd`/$BUILD/tmp/files-not-added-$target
    FILESUSED=`pwd`/$BUILD/tmp/files-used-$target
    NOTCHECKEDIN=`pwd`/$BUILD/tmp/files-not-checked-in-$target

    # Find which source files are used by the make command.  This allows us to not require
    # a perfectly clean git tree, but to make sure that all files that are used in the build
    # are committed so that the build is reproducible.
    #
    # It is done by running make in debug mode, and parsing the output for all files that it
    # considers targets.  This list is then filtered by removing all files that are outside the
    # source directory.

    echo -n "Checking which source files are used to make the target $target... " > /dev/stderr
    # Now find the source files used by make
    make -n --debug=v $target 2>/dev/null | grep 'Considering target file' | sed 's/ *Considering target file `\(.*\)'"'"'./\1/' | grep -v '^/' | grep -v 'build/' | sort > $FILESUSED
    cat $FILESUSED | wc -l > /dev/stderr

    # Dump these in a make friendly manner

    # Now extract a list of dirty submodules.  These will be scanned for dirty files that
    # are used in the build.
    ALLSUBMODULES=`git submodule status | awk '{ print $2; }'`
    DIRTYSUBMODULES=`git status --porcelain $ALLSUBMODULES | grep '^ M' | awk '{ print $2; }'`

    echo -n "Checking for uncommitted submodules... " > /dev/stderr
    NUMUNCOMMITEDSUBMODULES=`git status $ALLSUBMODULES | grep 'new commits' | wc -l`
    echo $NUMUNCOMMITEDSUBMODULES > /dev/stderr
    if [ $NUMUNCOMMITEDSUBMODULES -ne 0 ]; then
        echo $COLOR_ERROR"==== Uncommitted submodules $COLOR_RESET" > /dev/stderr
        echo -n $COLOR_RED > /dev/stderr
        git submodule status | awk '{ print $2; }' | xargs git status | grep 'modified:' > /dev/stderr
        echo -n $COLOR_RESET > /dev/stderr
        echo $COLOR_ERROR"==== Commit those submodules and try again $COLOR_RESET" > /dev/stderr
        DIRTY=1
    fi

    # function to check for non-added or uncommitted files used in the build in the given
    # submodule.  If the submodule passed is the empty string, then the main directory
    # is used.

    function check_submodule ()
    {
        sm=$1

        if [ -z "$sm" ]; then
            echo "doing the main dir" >> /dev/stderr
            SMFILESUSED=$FILESUSED
        else
            echo "doing submodule $sm" >> /dev/stderr
            SMFILESUSED=`pwd`/$BUILD/tmp/files-used-$target-submodule-$1
            pushd $sm > /dev/null
            echo -n "extracting build files used in dirty submodule $sm..." >> /dev/stderr
            cat $FILESUSED | grep "^$sm/" | sed "s!^$sm/!!g" | sort > $SMFILESUSED
        fi

        cat $SMFILESUSED | wc -l > /dev/stderr

        echo -n "  Checking for uncommitted files used in build... " > /dev/stderr
        cat $SMFILESUSED | xargs git status --porcelain -uno > $NOTCHECKEDIN

        NUMNOTCHECKEDIN=`cat $NOTCHECKEDIN | wc -l`
        echo $NUMNOTCHECKEDIN > /dev/stderr

        # Check for modified files that are used in the build.  These should be
        # committed as without them the build cannot be reproduced.
        if [ $NUMNOTCHECKEDIN -ne 0 ]; then
            echo $COLOR_ERROR"==== Uncommitted files in submodule $sm$COLOR_RESET" > /dev/stderr
            echo -n $COLOR_RED > /dev/stderr
            cat $SMFILESUSED | xargs git status -uno | grep 'modified:' > /dev/stderr
            echo -n $COLOR_RESET > /dev/stderr
            echo $COLOR_ERROR"==== Commit those files and try again $COLOR_RESET" > /dev/stderr

            # build is dirty; set the flag
            DIRTY=1
        fi

        # Check for files that haven't been added to git but are used in the build.
        # These should be committed as without them the build cannot be reproduced.
        echo -n "  Checking for unadded files used in build... " > /dev/stderr
        join <(git status --porcelain | grep '^??' | awk '{ print $2; }' | sort) <(sort $SMFILESUSED) > $UNADDEDFILES
        NUMNOTADDED=`cat $UNADDEDFILES | wc -l`
        echo $NUMNOTADDED > /dev/stderr

        if [ $NUMNOTADDED -ne 0 ]; then
            echo $COLOR_ERROR"==== Unadded files in submodule $sm$COLOR_RESET" > /dev/stderr
            echo -n $COLOR_RED > /dev/stderr
            cat $UNADDEDFILES | awk '{ print("        unadded:   ", $1); }' > /dev/stderr
            echo -n $COLOR_RESET > /dev/stderr
            echo $COLOR_ERROR"==== Add and commit those files and try again $COLOR_RESET" > /dev/stderr
            # build is dirty; set the flag
            DIRTY=1
        fi

        if [ -n "$sm" ]; then
            popd > /dev/null
        fi
    }

    # Go through the current directory ('') and the dirty submodules checking each of them
    # for being up to date.
    for sm in '' $DIRTYSUBMODULES; do
        check_submodule $sm
    done
fi

# Construct the tag from:
# - The branch
# - The user
# - The current git hash
# - A dirty indicator

branch=`git rev-parse --abbrev-ref HEAD`
user=`whoami`
hash=`git rev-parse HEAD`
ts=`date -u +%F-%T`
status=

if [ $DIRTY -ne 0 ]; then
    status="-dirty";
fi

hash=$user-$branch-$hash$status
echo "returning hash $hash" >> /dev/stderr
echo $hash

# Return the dirty status here
exit $DIRTY
