MAIN = notebook
SUBDIRS = entries python-notebooks
CLEANDIRS = $(SUBDIRS:%=clean-%)
DISTCLEANDIRS = $(SUBDIRS:%=distclean-%)

all: $(MAIN).pdf

$(MAIN).pdf: $(SUBDIRS) $(MAIN).tex
	latexmk

.PHONY: subdirs $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@

.PHONY: subdirs $(CLEANDIRS)
$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean

.PHONY: clean
clean: $(CLEANDIRS)
	latexmk -c
	rm -fv *.xdv
	rm -rfv _minted-*

.PHONY: distclean
distclean: clean $(DISTCLEANDIRS)
	rm -fv $(MAIN).pdf

.PHONY: subdirs $(DISTCLEANDIRS)
$(DISTCLEANDIRS):
	$(MAKE) -C $(@:distclean-%=%) distclean

