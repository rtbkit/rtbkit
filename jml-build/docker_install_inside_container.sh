#!/bin/bash

echo "installing from " $1

mkdir -p /opt
cp -r $1/* /opt
