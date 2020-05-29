INCFILE = numentries.txt
TEXFILES = $(wildcard *.tex)
ENTRIES = $(basename $(TEXFILES))

all: $(INCFILE)

$(INCFILE): $(TEXFILES)
	ls *.tex \
		| grep -e '^[0-9]\+' \
		| sort -n \
		| sed "s/\(.*\)\.tex/\\\\subfile{entries\/\1}/" \
		> $@

.PHONY: entries
entries: $(addsuffix .pdf, $(ENTRIES))

%.pdf: %.tex
	latexmk $<

.PHONY: clean
clean:
	latexmk -c
	rm -fv *.xdv
	rm -rfv _minted-*

.PHONY: distclean
distclean:
	rm -fv $(INCFILE)
	rm -fv *.pdf

