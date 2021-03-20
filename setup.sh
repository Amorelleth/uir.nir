#!/bin/bash

LPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "> LPT directory detected: ${LPTPATH}"
echo "> Setup lpt link in /usr/local/bin"
ln -s "${LPTPATH}/run.sh" "/usr/local/bin/lpt"

echo "> Pull required docker image"
docker pull deralusws/latex-papers-template-image:1.0

echo "> Installation successful"