FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    graphviz \
    lmodern \
    make \
    ghostscript \
    texlive-fonts-recommended \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-science \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
