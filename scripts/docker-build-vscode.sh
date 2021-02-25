DIR="$1"
DOC="$2"
ARG_DIR=."${DIR#*/latex-papers-template}"
ARG_DOC=."${DOC#*/latex-papers-template}"
echo $ARG_DIR
echo $ARG_DOC

${LATEX_PAPERS_ROOT_PATH}/scripts/docker-build-console.sh latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex -outdir="$ARG_DIR/out" "$ARG_DOC"