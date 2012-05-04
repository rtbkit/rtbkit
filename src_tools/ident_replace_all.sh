#!/bin/bash

if [ $# -lt 2 ] ;
    then
    echo "usage: " $0 "<regex> <replacement> [<directory>]";
    echo "(directory defaults to '.')";
    exit;
fi

REGEX=$1
REPLACEMENT=$2
DIR="."
if [ $# -eq 3 ];
    then
    DIR=$3
fi

FILES=`find $DIR -name "*.cc" -or -name "*.h" -or -name "*.i" | xargs grep -l "$REGEX"`

#echo "files = " $FILES

for file in $FILES ;
  do
  echo $file
  cp -f $file $file~
  cat $file~ | sed "s/$REGEX/$REPLACEMENT/g" > $file~~ && mv $file~~ $file
  chmod --reference=$file~ $file
done

