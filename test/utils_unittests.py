import unittest
from utils import _get_caller_directory, _get_config, get_database_config, get_workflow_config, get_lubw_config
from pathlib import Path


class UtilsTestCases(unittest.TestCase):

    def test_get_caller_directory_test_is_in_path(self):
        directory = _get_caller_directory(1)
        self.assertIn("test", str(directory))

    def test_get_config(self):
        path = Path(_get_caller_directory(1) / "test.yaml")
        config = _get_config(path)
        self.assertEqual(config["name"], "test")

    def test_get_database_config(self):
        database_config = get_database_config()
        self.assertEqual(database_config["url"], "test.com")

    def test_get_workflow_config(self):
        database_config = get_workflow_config()
        self.assertEqual(database_config["name"], "test")

    def test_get_lubw_config(self):
        database_config = get_lubw_config()
        self.assertEqual(database_config["name"], "test")


if __name__ == '__main__':
    unittest.main()
