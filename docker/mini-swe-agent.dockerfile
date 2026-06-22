FROM python:3.11-slim

ARG MINI_SWE_COMMIT=2caffc565474b8856a323ff163ffb7ab98d1ef02

RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    findutils \
    grep \
    sed \
    gawk \
    ripgrep \
    tree \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 miniagent

WORKDIR /opt
RUN curl -L https://github.com/SWE-agent/mini-swe-agent/archive/${MINI_SWE_COMMIT}.tar.gz \
    -o mini-swe-agent.tar.gz \
    && tar -xzf mini-swe-agent.tar.gz \
    && mv mini-swe-agent-${MINI_SWE_COMMIT} mini-swe-agent \
    && rm mini-swe-agent.tar.gz

WORKDIR /opt/mini-swe-agent
RUN pip install --upgrade pip \
    && pip install -e ".[full]" \
    && mini --help >/dev/null \
    && mini-extra --help >/dev/null

RUN mkdir -p /workspace/repo /workspace/results \
    && chown -R miniagent:miniagent /workspace

USER miniagent
WORKDIR /workspace/repo

CMD ["bash"]