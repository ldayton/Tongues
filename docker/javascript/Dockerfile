FROM node:21
RUN node --version | grep -q "v21\."
RUN npm install -g @biomejs/biome \
    && biome --version | grep -q "."
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install "~=3.12.0"
WORKDIR /workspace
