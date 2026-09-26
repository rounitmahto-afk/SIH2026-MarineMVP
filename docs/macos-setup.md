# macOS Development Setup

## Prerequisites

- Homebrew
- Python 3.14+
- Node.js 24+
- npm

Verify:

    python3 --version
    node --version
    npm --version
    brew --version

## Python environment

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements-backend.txt

Verify OpenCV:

    python -c "import cv2; print(cv2.__version__)"

## PostgreSQL and PostGIS

    brew install postgresql@17 postgis
    brew services start postgresql@17

For Apple Silicon Homebrew:

    echo 'export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"' >> ~/.zshrc
    source ~/.zshrc

Verify:

    psql --version
    pg_isready

Create the development database:

    psql postgres -c "CREATE ROLE sonar LOGIN PASSWORD 'sonar';"
    psql postgres -c "CREATE DATABASE sonar_mvp OWNER sonar;"
    psql -d sonar_mvp -c "CREATE EXTENSION postgis;"

## Environment

    cp .env.example .env

The development database URL is:

    DATABASE_URL=postgresql+psycopg://sonar:sonar@localhost:5432/sonar_mvp

Do not commit .env.

## Migrations

    alembic upgrade head
    alembic current

## Start the API

    uvicorn services.api.main:app --reload

Swagger UI:

    http://127.0.0.1:8000/docs

Health endpoints:

    /health/live
    /health/ready

## Notes

This guide is for local development. Keep credentials in .env local and do not commit secrets.
