#!/bin/bash
for f in *.stl
do
	for g in *.stl
	do
 		./compute_OTM $f $g
		h="$(./checkmorph)" 
		echo "$f $g $h" >> bench.txt
	done
done
