#!/bin/sh

sed -i "s/___currentdir___/$( pwd | sed 's/\//\\\//g' )/" exec.pbs
