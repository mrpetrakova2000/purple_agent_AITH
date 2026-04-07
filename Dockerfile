FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY --chown=agent:agent pyproject.toml uv.lock README.md ./
COPY --chown=agent:agent src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv lock && \
    uv sync --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009