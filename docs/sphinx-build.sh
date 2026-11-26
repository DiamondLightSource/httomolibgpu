#!/usr/bin/env bash
# Generate API pages and build documentation without running notebook cells.
set -euo pipefail
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPHINX_APIDOC_OPTIONS='members,show-inheritance,undoc-members'
python -m sphinx.ext.apidoc -feT -t "$DOCS_DIR/source/_templates" \
    -o "$DOCS_DIR/source/api" "$DOCS_DIR/../httomolibgpu"
OUTPUT_DIR="${1:-$DOCS_DIR/build}"
# Sphinx viewcode otherwise keeps source HTML newer than the Python module,
# even when API coverage changes. Remove only its generated source-page cache.
python -c 'import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)' "$OUTPUT_DIR/_modules"
# Refresh cached API/source links and rewrite pages after navigation edits.
python -m sphinx -E -a -b html -W --keep-going "$DOCS_DIR/source" "$OUTPUT_DIR"
