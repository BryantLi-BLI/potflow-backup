"""
Settings for potflow.
"""

from pathlib import Path

from pydantic import BaseSettings, Field, root_validator

_DEFAULT_CONFIG_FILE_PATH = Path("~/.potflow.yaml").expanduser().as_posix()


class PotflowSettings(BaseSettings):
    """
    Settings for Potflow.

    The default way to modify the seetings is to modify `~/.potflow.yaml.

    Alternatively, the environment variable POTFLOW_CONFIG_FILE can be set to point
    to a yaml file with the settings.

    In addition, the variables can be modified directly through environment variables by
    using the "POTFLOW" prefix. E..g., POTFLOW_SCRATCH_DIR = /path/to/scratch.
    """

    # TODO, check how the environment variables work

    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH,
        description="File from which to load alternative defaults.",
    )

    # general settings
    SYMPREC: float = Field(
        0.1, description="Symmetry precision for spglib symmetry finding."
    )

    class Config:
        """Pydantic config settings."""

        env_prefix = "potflow_"

    # TODO check how this part works
    @root_validator(pre=True)
    def load_default_settings(cls, values):
        """
        Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built in defaults.

        This allows setting of the config file path through environment variables.
        """
        from monty.serialization import loadfn

        config_file_path: str = values.get("CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)

        new_values = {}
        if Path(config_file_path).exists():
            new_values.update(loadfn(config_file_path))

        new_values.update(values)

        return new_values
