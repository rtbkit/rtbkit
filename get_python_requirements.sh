#!/bin/bash

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
    echo "Error: missing argument"
    echo
    usage
    exit 1
fi

# Loop throughthe requirements
for requirement in `cat $req_file`
do
    cmd="$pip_command install $requirement"
    echo $cmd
    $cmd
done	

