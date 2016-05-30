#!/bin/bash

set -e
set -x

echo "installing from " $1

cd $1
cp -av * /

shift
$*
