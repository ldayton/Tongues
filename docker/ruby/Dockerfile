FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y ruby ruby-dev build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && ruby --version | grep -q "3\." \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN gem install rubocop --no-document \
    && rubocop --version | grep -q "."
RUN uv python install "~=3.12.0"
ARG JUST_VERSION=1.46.0
RUN curl -sSfL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C /usr/local/bin just
WORKDIR /workspace
