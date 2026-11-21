# Agents

**Using Pauli Cones:** run `pauli-cones guide`, or read [the agent guide](src/pauli_cones/AGENT_GUIDE.md). It covers when to use the package, the five commands, a decision recipe, how to read results and how to verify them. `pauli-cones capabilities` and `pauli-cones schema` describe the supported gates, limits and request format as JSON.

**Changing Pauli Cones:**
- Read the [README](README.md), [architecture](docs/architecture.md) and [contribution guide](CONTRIBUTING.md).
- Run `python -m pytest -q` after building the extension with `python setup.py build_ext --inplace`.
- Keep the command line JSON-only, with a `hint` on every error.
- Keep the agent guide in step with any change to commands or outputs.
