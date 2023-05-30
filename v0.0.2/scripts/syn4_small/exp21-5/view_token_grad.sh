#!/usr/bin/env bash
if [ -z "$2" ]
then
  METHOD=mean
else
  METHOD=$2
fi
cat $1 | sortpy --n 0 --i 0 --r --aggregate 2,3:$METHOD