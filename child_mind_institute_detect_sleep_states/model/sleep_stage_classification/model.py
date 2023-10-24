import pathlib
import sys
project_root_path = pathlib.Path(__file__).parent.parent.parent.parent

__all__ = ["CharGRULSTM"]

sys.path.append(str(project_root_path / "lib"))

from sleep_stage_classification.code.LSTM import CharGRULSTM  # noqa

sys.path.pop(-1)
