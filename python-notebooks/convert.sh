#!/bin/sh

f="entropy"

pandoc --no-highlight --filter ./minted.py "${f}.md" \
  -o "${f}.tex" \
  -N \
  --verbose

