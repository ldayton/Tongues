FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y perl perltidy curl \
    && rm -rf /var/lib/apt/lists/* \
    && perl -v | grep -q "v5\.38" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install "~=3.14"
WORKDIR /workspace
