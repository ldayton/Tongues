FROM node:21
RUN npm install -g typescript@5.3 tsx @biomejs/biome \
    && tsc --version | grep -q "5\.3"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install "~=3.14"
WORKDIR /workspace
