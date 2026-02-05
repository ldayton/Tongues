FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y lua5.4 curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && lua -v 2>&1 | grep -q "5\.4"
RUN ARCH=$(dpkg --print-architecture) \
    && curl -sSL -o /usr/local/bin/stylua \
       "https://github.com/JohnnyMorganz/StyLua/releases/download/v2.0.2/stylua-linux-${ARCH}" \
    && chmod +x /usr/local/bin/stylua \
    && stylua --version | grep -q "."
WORKDIR /workspace
