IMAGE=deralusws/latex-papers-template-image:1.0
exec docker run --rm -i --user="$(id -u):$(id -g)" --net=none -v "$LATEX_PAPERS_ROOT_PATH/":/data "$IMAGE" "$@"