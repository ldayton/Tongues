FROM swift:6.0
RUN swift --version | head -1 | grep -q "6\."
RUN apt-get update && apt-get install -y curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "arm64" ]; then SUFFIX="_aarch64"; else SUFFIX=""; fi \
    && curl -sSL "https://github.com/nicklockwood/SwiftFormat/releases/download/0.59.1/swiftformat_linux${SUFFIX}.zip" -o /tmp/sf.zip \
    && unzip -q /tmp/sf.zip -d /tmp \
    && mv /tmp/swiftformat_linux* /usr/local/bin/swiftformat \
    && chmod +x /usr/local/bin/swiftformat \
    && rm /tmp/sf.zip \
    && swiftformat --version
RUN uv python install "~=3.12.0"
WORKDIR /workspace
