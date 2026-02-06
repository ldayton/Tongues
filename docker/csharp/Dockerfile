FROM mcr.microsoft.com/dotnet/sdk:8.0
RUN dotnet --version | grep -q "8\."
RUN apt-get update && apt-get install -y curl mono-mcs \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/root/.dotnet/tools:${PATH}"
RUN dotnet tool install -g csharpier \
    && csharpier --version | grep -q "."
RUN uv python install "~=3.12.0"
WORKDIR /workspace
