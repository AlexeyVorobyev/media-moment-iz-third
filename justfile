set shell := ["bash", "-c"]

init:
    python3.12 -m venv venv && \
    source ./venv/bin/activate && \
    pip install poetry && \
    poetry install

source:
    source ./venv/bin/activate

run target:
    source ./venv/bin/activate && \
    python3 {{target}}.py