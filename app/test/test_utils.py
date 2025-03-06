from app.utils import _get_caller_directory, get_config


def test_get_caller_directory_test_is_in_path():
    directory = _get_caller_directory(1)
    assert "test" in str(directory)


def test_get_config():
    config = get_config("ressources/test.yaml")
    assert config["name"] == "test"


def test_get_database_config():
    database_config = get_config("ressources/database_config.yaml")
    assert database_config["url"] == "test.com"


def test_get_workflow_config():
    database_config = get_config("ressources/workflow_config.yaml")
    assert database_config["name"] == "test"

