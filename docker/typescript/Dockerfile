FROM node:21
RUN npm install -g typescript@5.3 tsx @biomejs/biome \
    && tsc --version | grep -q "5\.3"
WORKDIR /workspace
