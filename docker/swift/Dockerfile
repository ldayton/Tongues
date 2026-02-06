FROM swift:5.9
RUN swift --version | head -1 | grep -q "5\.9"
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN git clone --depth 1 --branch 510.1.0 https://github.com/apple/swift-format.git /tmp/swift-format \
    && cd /tmp/swift-format && swift build -c release \
    && cp /tmp/swift-format/.build/release/swift-format /usr/local/bin/ \
    && rm -rf /tmp/swift-format \
    && swift-format --version | grep -q "."
RUN uv python install "~=3.12.0"
WORKDIR /workspace
