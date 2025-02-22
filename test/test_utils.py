from utils import _get_caller_directory, _get_config, get_database_config, get_workflow_config, get_lubw_config
from pathlib import Path


def test_get_caller_directory_test_is_in_path():
    directory = _get_caller_directory(1)
    assert "test" in str(directory)


def test_get_config():
    path = Path(_get_caller_directory(1) / "test.yaml")
    config = _get_config(path)
    assert config["name"] == "test"


def test_get_database_config():
    database_config = get_database_config()
    assert database_config["url"] == "test.com"


def test_get_workflow_config():
    database_config = get_workflow_config()
    assert database_config["name"] == "test"


def test_get_lubw_config():
    database_config = get_lubw_config()
    assert database_config["name"] == "test"
