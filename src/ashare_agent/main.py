from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from ashare_agent.api.routes import router
from ashare_agent.container import AppContainer
from ashare_agent.core.config import Settings


PROJECT_ROOT_ENV = "ASHARE_AGENT_PROJECT_ROOT"


def create_app(
    container: AppContainer | None = None,
    *,
    project_root: Path | None = None,
) -> FastAPI:
    if container is not None and project_root is not None:
        raise ValueError("Pass either container or project_root, not both")
    effective_container = container or AppContainer.build(
        Settings.from_env(project_root)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        effective_container.repository.initialize()
        app.state.container = effective_container
        yield
        effective_container.task_runner.shutdown()

    application = FastAPI(
        title="Ashare Agent XJ",
        version="0.1.0",
        description=(
            "Manual portfolio ingestion and asynchronous AI-assisted decisions. "
            "The service returns advisory results and never submits broker orders."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


def create_app_from_environment() -> FastAPI:
    raw_project_root = os.getenv(PROJECT_ROOT_ENV, "").strip()
    project_root = Path(raw_project_root).resolve() if raw_project_root else None
    return create_app(project_root=project_root)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = create_app_from_environment()
