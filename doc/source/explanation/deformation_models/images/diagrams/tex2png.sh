#!/bin/bash

pdflatex linear_general.tex
convert -density 600 linear_general.pdf linear_general.png

pdflatex linear_momentum.tex
convert -density 600 linear_momentum.pdf linear_momentum.png

pdflatex linear_code.tex
convert -density 600 linear_code.pdf linear_code.png

pdflatex linear_displacement.tex
convert -density 600 linear_displacement.pdf linear_displacement.png
