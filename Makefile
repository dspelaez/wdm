# vim:ft=make
#
# =====================
# Makefile
# Daniel Santiago 
# 2018-06-13
# =====================

all:
	pip install .

requirements:
	pip install -r requirements.txt .

clean:
	rm -rf *.egg-info dist build
