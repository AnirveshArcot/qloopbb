import os
import tempfile
import unittest
from pathlib import Path

from qloopbb.env import load_dotenv, strip_env_value


class EnvTests(unittest.TestCase):
    def test_strip_env_value_removes_matching_quotes(self) -> None:
        self.assertEqual(strip_env_value('"secret"'), "secret")
        self.assertEqual(strip_env_value("'secret'"), "secret")
        self.assertEqual(strip_env_value("secret"), "secret")

    def test_load_dotenv_sets_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "GEMINI_API_KEY=test-key",
                        "QUOTED=\"quoted-value\"",
                        "export EXPORTED=exported-value",
                    ]
                ),
                encoding="utf-8",
            )

            previous_key = os.environ.pop("GEMINI_API_KEY", None)
            previous_quoted = os.environ.pop("QUOTED", None)
            previous_exported = os.environ.pop("EXPORTED", None)
            try:
                load_dotenv(path)

                self.assertEqual(os.environ["GEMINI_API_KEY"], "test-key")
                self.assertEqual(os.environ["QUOTED"], "quoted-value")
                self.assertEqual(os.environ["EXPORTED"], "exported-value")
            finally:
                restore_env("GEMINI_API_KEY", previous_key)
                restore_env("QUOTED", previous_quoted)
                restore_env("EXPORTED", previous_exported)

    def test_load_dotenv_does_not_override_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("GEMINI_API_KEY=file-key", encoding="utf-8")

            previous_key = os.environ.get("GEMINI_API_KEY")
            os.environ["GEMINI_API_KEY"] = "shell-key"
            try:
                load_dotenv(path)

                self.assertEqual(os.environ["GEMINI_API_KEY"], "shell-key")
            finally:
                restore_env("GEMINI_API_KEY", previous_key)


def restore_env(key: str, value) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
