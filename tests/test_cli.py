import unittest
from unittest import mock
from unittest.mock import patch

from typer.testing import CliRunner

from corgi import cli

class TestCLI(unittest.TestCase):
    def setUp(self):  #special function that runs before any test
        self.runner = CliRunner()

    def test_version(self):
        result = self.runner.invoke(cli.app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout

    @patch("typer.launch")
    def test_repo(self, mock):
        result = self.runner.invoke(cli.app, ["repo"])
        mock.assert_called_with("https://gitlab.unimelb.edu.au/mdap/corgi")