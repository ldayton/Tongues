FROM eclipse-temurin:21
RUN java --version | head -1 | grep -q "21\."
RUN apt-get update && apt-get install -y curl \
    && curl -sSL -o /usr/local/lib/google-java-format.jar \
       https://github.com/google/google-java-format/releases/download/v1.25.2/google-java-format-1.25.2-all-deps.jar \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
