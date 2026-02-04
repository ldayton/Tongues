FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y perl perltidy \
    && rm -rf /var/lib/apt/lists/* \
    && perl -v | grep -q "v5\.38"
WORKDIR /workspace
