FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "arm64" ]; then ZIG_ARCH="aarch64"; else ZIG_ARCH="x86_64"; fi \
    && curl -sSL "https://ziglang.org/download/0.15.2/zig-linux-${ZIG_ARCH}-0.15.2.tar.xz" -o /tmp/zig.tar.xz \
    && tar -xf /tmp/zig.tar.xz -C /opt \
    && mv /opt/zig-linux-* /opt/zig \
    && rm /tmp/zig.tar.xz \
    && /opt/zig/zig version | grep -q "0\.15\.2" \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/opt/zig:${PATH}"
RUN uv python install "~=3.12.0"
WORKDIR /workspace
