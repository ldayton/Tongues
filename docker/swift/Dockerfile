FROM swift:5.9
RUN swift --version | head -1 | grep -q "5\.9"
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install "~=3.12.0"
WORKDIR /workspace
