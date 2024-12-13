# Установка проекта

## Linux и наличие утилиты just
```bash
just init
```

## Linux
```bash
python3.13 -m venv venv && \
    source ./venv/bin/activate && \
    pip install poetry && \
    poetry install
```

## Windows
```shell
python3 -m venv venv
./venv/Scripts/activate
pip install poetry
poetry install
```