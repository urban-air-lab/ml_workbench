from app.get_config import _get_caller_directory, get_config


def test_get_caller_directory_test_is_in_path():
    directory = _get_caller_directory(1)
    assert "test" in str(directory)


def test_get_config():
    config = get_config("ressources/test.yaml")
    assert config["name"] == "test"


