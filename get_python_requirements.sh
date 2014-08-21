#!/bin/bash

# Usage message
usage () {
    echo "Usage: $0 [options]"
    echo
    echo "Options"
    echo "    -c arg     pip command"
    echo "    -r arg     file with thei python requirements"
    echo
}

# Get the options
while getopts c:r: option
do
case "${option}"
    in
        c) pip_command=${OPTARG};;
        r) req_file=${OPTARG};;
        ?) echo "Error: unknown option: - ${OPTARG}"
           usage
           exit 1
           ;;
    esac
done

# Check for missing variables
if [[ -z "$pip_command" || -z "$req_file" ]]; then
    echo
    echo "Error: missing argument"
    echo
    usage
    exit 1
fi

# Check for missing file
if [[ ! -f "$req_file" ]]; then
    echo
    echo "Error: missing file $req_file"
    echo
    usage
    exit 1
fi

# Remove commented and empty lines & loop through the requirements
grep -vE '^#|^$' $req_file | while read line
do
    cmd="$pip_command install $line"
    echo $cmd
    $cmd
done
