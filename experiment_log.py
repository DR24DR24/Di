"""
experiment_log.py — lightweight experiment log directory management.

Purpose
-------
This module provides a decorator for assigning a deterministic directory to
each experiment configuration.

The decorator does NOT serialize experiment results. The decorated function
decides which files to write (CSV, JSON, NumPy, PyTorch checkpoints, etc.).
It receives only the directory path through its ``log_path`` argument.

Main interface
--------------
After decorating a function, three helper attributes are available:

1. ``function.log_params``
   Tuple of parameter names included in the experiment identity/hash.

2. ``function.get_log_path(*args, **kwargs)``
   Computes the expected directory for the supplied function arguments.
   It does not create the directory and does not run the decorated function.

3. ``function.read_logs(*args, **kwargs)``
   Checks whether the expected directory already exists. Returns its Path
   if it exists, otherwise returns None. It does not parse or read files.

When the decorated function is actually called, the wrapper:
- binds arguments using the original function signature;
- applies default values;
- selects only the parameters listed in ``params``;
- computes a deterministic hash from those selected values;
- creates the corresponding directory;
- passes that directory to the function as ``log_path``.

Basic usage
-----------
    @experiment_log(
        params=["dataset", "model", "seed"],
        root="logs",
    )
    def train_model(
        dataset,
        model,
        seed=42,
        epochs=50,
        log_path=None,
    ):
        output_file = log_path / "training.csv"

        # Perform training and save results in your preferred format.
        # The function itself owns the file format and serialization.

    # These two calls identify the same configuration because seed defaults
    # to 42 in both cases:
    path1 = train_model.get_log_path("CIFAR10", "ResNet18")
    path2 = train_model.get_log_path(
        dataset="CIFAR10",
        model="ResNet18",
        seed=42,
    )
    assert path1 == path2

    # The path is computed only; no directory is created:
    expected_path = train_model.get_log_path(
        "CIFAR10", "ResNet18", seed=123
    )

    # Check whether results for this configuration already exist:
    existing_path = train_model.read_logs(
        "CIFAR10", "ResNet18", seed=123
    )
    if existing_path is not None:
        print("Found experiment:", existing_path)
    else:
        print("No saved results for this configuration yet.")

    # Calling the procedure creates its log directory and passes it as
    # the log_path argument:
    train_model("CIFAR10", "ResNet18", epochs=100)

Parameter selection
-------------------
Only names listed in ``params`` affect the hash. Other function arguments
(e.g. ``epochs`` in the example) do not affect the directory name.

This is useful when several procedures share a large configuration but
depend on different subsets of it. Each procedure can declare its own
identity parameters:

    @experiment_log(params=["dataset", "model", "seed"])
    def train_model(dataset, model, seed=42, log_path=None):
        ...

    @experiment_log(params=["dataset", "model", "n_components"])
    def calculate_projection(dataset, model, n_components=200,
                             log_path=None):
        ...

Argument handling
-----------------
``get_log_path`` and the decorated function accept positional and keyword
arguments. Python's normal signature binding rules apply, including default
values. Missing required arguments and unexpected arguments raise the same
kind of TypeError as they would for the original function.

Requirements and limitations
----------------------------
- The decorated function must declare a ``log_path`` parameter.
- Names in ``params`` must exist in the decorated function signature.
- Parameter values should be JSON-serializable for stable hashing.
  The current ``default=str`` fallback is convenient for simple objects,
  but custom objects may need explicit normalization to guarantee identical
  hashes across environments or software versions.
- If two calls have the same selected parameter values, they share a
  directory. If you need to keep repeated runs separate, include a run ID
  or another distinguishing parameter in ``params``.
- The hash is shortened to 12 hexadecimal characters. This is generally
  practical for experiment directories, but it is not a mathematical
  guarantee against collisions.
"""

import json
import hashlib
import inspect

from pathlib import Path
from functools import wraps


def make_hash(params):
    """Return a deterministic hash for a parameter dictionary."""
    data = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def experiment_log(params, root="logs"):
    """
    Decorator that manages experiment log directories.

    Args:
        params: Names of function parameters included in the experiment hash.
        root: Root directory for all experiment logs.

    The decorated function must declare a `log_path` parameter.
    """

    params = tuple(params)

    def decorator(func):
        signature = inspect.signature(func)
        root_path = Path(root)

        # Validate parameter names at decoration time.
        unknown = set(params) - set(signature.parameters)
        if unknown:
            raise ValueError(
                f"Unknown parameters for {func.__name__}: {sorted(unknown)}"
            )

        if "log_path" not in signature.parameters:
            raise ValueError(
                f"{func.__name__} must declare a 'log_path' parameter"
            )

        def select_params(*args, **kwargs):
            """
            Select the identity parameters from a (possibly partial) call.

            Only the parameters listed in ``params`` have to be supplied;
            any other required argument of the decorated function (data,
            loaders, devices, ...) may be omitted. Defaults are applied as in
            a normal call. A missing identity parameter raises TypeError.
            """
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            missing = [name for name in params if name not in bound.arguments]
            if missing:
                raise TypeError(
                    f"{func.__name__}: missing identity parameters {missing}"
                )
            return {name: bound.arguments[name] for name in params}

        def get_log_path(*args, **kwargs):
            """Compute the log directory without creating it."""
            selected = select_params(*args, **kwargs)
            hash_id = make_hash(selected)
            return root_path / func.__name__ / hash_id

        def read_logs(*args, **kwargs):
            """
            Return the existing log directory, or None if it does not exist.

            This function does not read individual files; their formats are
            owned by the decorated procedure.
            """
            path = get_log_path(*args, **kwargs)
            return path if path.is_dir() else None

        @wraps(func)
        def wrapper(*args, **kwargs):
            # A real call is strict: all required arguments must be given.
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            selected = {
                name: bound.arguments[name]
                for name in params
            }
            hash_id = make_hash(selected)

            log_path = root_path / func.__name__ / hash_id
            log_path.mkdir(parents=True, exist_ok=True)

            # Supply the directory to the procedure.
            bound.arguments["log_path"] = log_path

            return func(*bound.args, **bound.kwargs)

        # Public helper attributes attached to the decorated function.
        wrapper.log_params = params
        wrapper.get_log_path = get_log_path
        wrapper.read_logs = read_logs

        return wrapper

    return decorator


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Example
    # ---------------------------------------------------------------------

    @experiment_log(
        params=["dataset", "model", "seed"],
        root="logs",
    )
    def train_model(
        dataset,
        model,
        seed=42,
        epochs=50,
        log_path=None,
    ):
        """
        Example procedure.

        The procedure owns the format and content of its output files.
        """
        results_file = log_path / "training.csv"

        # Replace this example with the actual training and saving code.
        # For example:
        #
        # import pandas as pd
        # results = ...
        # results.to_csv(results_file, index=False)

        print(f"Saving results to: {results_file}")


        # Parameters participating in the hash:
        print("Hash parameters:", train_model.log_params)

        # Compute path without creating the directory:
        path = train_model.get_log_path("CIFAR10", "ResNet18")
        print("Expected log path:", path)

        # Check whether the directory already exists:
        existing_path = train_model.read_logs(
            dataset="CIFAR10",
            model="ResNet18",
        )
        print("Existing log path:", existing_path)

        # Run the procedure; this creates the directory:
        # train_model("CIFAR10", "ResNet18", epochs=100)
