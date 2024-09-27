from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import List, Optional, Union

__all__ = ["Process", "execute_process", "execute_processes_concurently"]

CAPTURING_ENCODING = "utf8"


@dataclass(frozen=True)
class Process:
    """
    By default `encoding` is "utf8".
    Uses shell if `executable` is not provided.

    See:
    * https://docs.python.org/3/library/subprocess.html#subprocess.run
    * https://docs.python.org/3/library/subprocess.html#subprocess.Popen
    """
    args: List[Union[Path, str]]
    input_data: Optional[str] = None
    executable: Optional[Path] = None
    timeout: Optional[float] = None
    encoding: Optional[str] = CAPTURING_ENCODING


def execute_process(process: Process) -> CompletedProcess:
    # TODO: improve docs
    """Simple function for running software.
    
    >>> execute_process(args=['ls', '-l'])
    >>> execute_process(args=['/path/to/software'], stdin=input_file_data)
    """
    return run(
        executable=process.executable,
        stdin=process.input_data,
        args=process.args,
        timeout=process.timeout,
        encoding=process.encoding,
        capture_output=True,
    )