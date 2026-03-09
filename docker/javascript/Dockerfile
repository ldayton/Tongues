FROM node:21
RUN node --version | grep -q "v21\."
RUN npm install -g @biomejs/biome \
    && biome --version | grep -q "."
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install "~=3.12.0"
ARG JUST_VERSION=1.46.0
RUN curl -sSfL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C /usr/local/bin just
WORKDIR /workspace
