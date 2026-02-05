FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ondrej/php \
    && apt-get update && apt-get install -y php8.3-cli php8.3-mbstring php8.3-intl curl \
    && rm -rf /var/lib/apt/lists/* \
    && php --version | head -1 | grep -q "8\.3"
RUN curl -sSL -o /usr/local/bin/php-cs-fixer \
       https://cs.symfony.com/download/php-cs-fixer-v3.phar \
    && chmod +x /usr/local/bin/php-cs-fixer \
    && php-cs-fixer --version | grep -q "."
WORKDIR /workspace
