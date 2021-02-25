FROM ubuntu:focal
ENV DEBIAN_FRONTEND noninteractive

RUN echo "deb http://archive.ubuntu.com/ubuntu/ trusty multiverse" | tee -a /etc/apt/sources.list.d/multiverse.list && \
   echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN apt update && \
   apt install -y wget git make apt-transport-https unzip && \
   apt install -y texlive-base texlive-latex-extra texlive-xetex texlive-lang-cyrillic latexmk fonts-linuxlibertine && \
   apt install -y --reinstall ttf-mscorefonts-installer

WORKDIR /data
VOLUME ["/data"]