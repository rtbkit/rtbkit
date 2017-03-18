#!/bin/bash

make compile -j 4 NODEJS_ENABLED=0
make test NODEJS_ENABLED=0
