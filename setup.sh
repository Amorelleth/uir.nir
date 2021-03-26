#!/bin/bash

LPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "> LPT directory detected: ${LPTPATH}"
LPTLINK=lpt

choose_lpt() {
    while :
    do
        read -p "< Enter alias to automation tool of this local repository (default is 'lpt'): " TLPTLINK
        if [ -z "${TLPTLINK}" ]; then
            TLPTLINK=lpt
        fi
        which "${TLPTLINK}" >> /dev/null
        if [ $? -ne 0 ]; then
            LPTLINK=${TLPTLINK}
            return
        else
            echo "> Alias '${TLPTLINK}' has already in use in system, choose another one"
        fi
    done
}

choose_lpt

echo "> Setup ${LPTLINK} link in /usr/local/bin"
ln -s "${LPTPATH}/run.sh" "/usr/local/bin/${LPTLINK}"

echo "> Pull required docker image"
docker pull deralusws/latex-papers-template-image:1.0

echo "> Installation successful"