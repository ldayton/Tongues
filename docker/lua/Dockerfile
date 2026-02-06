FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y lua5.4 curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && lua -v 2>&1 | grep -q "5\.4" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then STYLUA_ARCH="x86_64"; else STYLUA_ARCH="aarch64"; fi \
    && curl -sSL -o /tmp/stylua.zip \
       "https://github.com/JohnnyMorganz/StyLua/releases/download/v2.0.2/stylua-linux-${STYLUA_ARCH}.zip" \
    && unzip -o /tmp/stylua.zip -d /usr/local/bin \
    && rm /tmp/stylua.zip \
    && chmod +x /usr/local/bin/stylua \
    && stylua --version | grep -q "."
RUN uv python install "~=3.12.0"
WORKDIR /workspace
