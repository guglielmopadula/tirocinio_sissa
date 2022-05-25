#!/bin/bash
python splitfile.py
cut -c 3- table.txt > table2.txt && mv table2.txt table.txt
tail -n +3 points.txt > points2.txt && mv points2.txt points.txt
