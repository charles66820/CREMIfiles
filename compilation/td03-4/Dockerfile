FROM openjdk:11-jdk-buster

RUN apt-get -y update && apt-get -y install build-essential manpages-dev m4 ant

RUN wget http://ftp.gnu.org/gnu/bison/bison-3.7.tar.gz && \
  tar -zxvf bison-3.7.tar.gz && \
  cd bison-3.7/ && \
  ./configure && \
  make && \
  make install

COPY . /usr/src/app
WORKDIR /usr/src/app

CMD ["make"]