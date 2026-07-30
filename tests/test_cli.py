from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from ashare_agent import cli
from ashare_agent import main as application_module


def test_serve_forwards_project_root_while_uvicorn_is_running(
    tmp_path,
    monkeypatch,
):
    project_root = tmp_path / "configured-project"
    project_root.mkdir()
    captured: dict[str, object] = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs
        captured["project_root"] = os.environ.get(
            application_module.PROJECT_ROOT_ENV
        )

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    monkeypatch.setenv(application_module.PROJECT_ROOT_ENV, "previous-value")

    exit_code = cli.main(
        [
            "--project-root",
            str(project_root),
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--reload",
        ]
    )

    assert exit_code == 0
    assert captured == {
        "app": "ashare_agent.main:app",
        "kwargs": {
            "host": "0.0.0.0",
            "port": 8765,
            "reload": True,
        },
        "project_root": str(project_root.resolve()),
    }
    assert os.environ[application_module.PROJECT_ROOT_ENV] == "previous-value"


def test_environment_app_factory_builds_container_from_forwarded_root(
    tmp_path,
    monkeypatch,
):
    project_root = (tmp_path / "runtime-project").resolve()
    project_root.mkdir()
    settings = object()
    container = object()
    from_env = Mock(return_value=settings)
    build = Mock(return_value=container)
    monkeypatch.setattr(application_module.Settings, "from_env", from_env)
    monkeypatch.setattr(application_module.AppContainer, "build", build)
    monkeypatch.setenv(
        application_module.PROJECT_ROOT_ENV,
        str(project_root),
    )

    application = application_module.create_app_from_environment()

    from_env.assert_called_once_with(project_root)
    build.assert_called_once_with(settings)
    assert application.title == "Ashare Agent XJ"
