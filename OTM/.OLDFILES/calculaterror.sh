#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
num=1000
rm error.csv
for i in {1..100}
do
	ans=$((num * i))
	./compute_OTM cube.meshb Parallelepiped.meshb nb_pts=$ans
	python extract.py
	vorpastat cube.meshb ext2.tet6 > temp.txt
	var1=$(python geterror.py)
	vorpastat Parallelepiped.meshb ext1.tet6>temp.txt
	var2=$(python geterror.py)
	echo "$ans,$var1,$var2">> error.csv
done
