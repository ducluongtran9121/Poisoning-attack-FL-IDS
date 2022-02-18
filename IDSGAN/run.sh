#!/bin/sh

FILE=[item1 item2 item3]
FILE[0]="BlackBox_IDS.py"
FILE[1]="IDS_WGAN.py"
FILE[2]="generate_attacktraffic.py"

for f in ${FILE[@]}; do
    eval "python ${f}"
done