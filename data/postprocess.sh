#!/bin/sh

# input path
INPUT=$1

# output path
OUTPUT=$2

# restore subword units to original segmentation
sed -r 's/(@@ )|(@@ ?$)//g' ${INPUT} > ${OUTPUT}
