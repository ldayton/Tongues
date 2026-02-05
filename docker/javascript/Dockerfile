FROM node:21
RUN node --version | grep -q "v21\."
RUN npm install -g @biomejs/biome \
    && biome --version | grep -q "."
WORKDIR /workspace
