#!/bin/bash
INPUTS=configs

DIR=../results
rm -rf ${DIR}

for FILE in $(find ${INPUTS} -type f); do
    echo "Processing config file ${FILE}"
    python simulate.py ${FILE}
done