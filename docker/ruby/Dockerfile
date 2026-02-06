FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y ruby curl \
    && rm -rf /var/lib/apt/lists/* \
    && ruby --version | grep -q "3\." \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN gem install rubocop --no-document \
    && rubocop --version | grep -q "."
RUN uv python install "~=3.14"
WORKDIR /workspace
