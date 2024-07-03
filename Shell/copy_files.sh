#!/bin/bash

# Check if enough arguments are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Get source and destination directories from command line arguments
SOURCE_DIR="$1"
DEST_DIR="$2"

# Create the destination directory if it does not exist
if [[ ! -e $DEST_DIR ]]; then
    mkdir -p $DEST_DIR
elif [[ -d $DEST_DIR ]]; then
    echo "Directory $DEST_DIR already exists"
else
    echo "$DEST_DIR already exists but is not a directory" 1>&2
    exit 1
fi

# Iterate over .mzML files in the source directory
for file in "$SOURCE_DIR"/*.mzML; do
    # Check if the filename does not contain a specified string
    if [[ ! $(basename "$file") =~ 150min|135min|120min|105min|90min|75min|60min ]]; then
        # Copy the file to the destination directory
        cp "$file" "$DEST_DIR"
        echo "Copied $(basename "$file") to $DEST_DIR"
    fi
done
