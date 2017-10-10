SUBDIRS := AEBEM2 AEBEM3 AEBEMA AIBEM2 AIBEM3 AIBEMA

# Disabled specific program that rely on NAG function f02akf
# AMBEM2 AMBEM3

all: gsl/lib/libgsl.a gsl/lib/libfgsl.a $(SUBDIRS)

$(SUBDIRS):
	make -C $@

gsl/lib/libgsl.a:
	./install_gsl.sh

gsl/lib/libfgsl.a:
	./install_fgsl.sh

.PHONY: all $(SUBDIRS)