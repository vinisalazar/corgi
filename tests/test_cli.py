import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from corgi import cli

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(cli.app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout