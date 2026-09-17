From the repository root, install the package and its dependencies, then run
the suite (Python 3.10 or newer):

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

These commands work on Windows, macOS, and Linux. No additional test dependencies
or `PYTHONPATH` configuration are needed.

Tests generate their own images and use temporary directories. SIFT alignment
and PNG output use real OpenCV operations; video encoding is mocked to keep the
suite independent of system codec availability. Actual MP4 playback is not tested.
