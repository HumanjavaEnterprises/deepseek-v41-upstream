# Contributing

Thanks for helping. Here is where each kind of report goes.

| You found | Where it goes |
|---|---|
| The repack tool or BF16 build tools fail, or a command in RECIPE.md doesn't work | [Tool bug](https://github.com/HumanjavaEnterprises/deepseek-v41-upstream/issues/new?template=bug.yml) |
| The model loads but outputs are wrong, garbled, or slower than the EVALS figures | [Model output problem](https://github.com/HumanjavaEnterprises/deepseek-v41-upstream/issues/new?template=output.yml) |
| A typo, an unclear step, a missing setup | [Docs fix](https://github.com/HumanjavaEnterprises/deepseek-v41-upstream/issues/new?template=docs.yml), or open a pull request |
| A security problem (tampered weights, unexpected embedded content) | **Privately**, see [SECURITY.md](SECURITY.md) |

## Pull requests

Welcome for the tools, tests and docs. Keep changes small and focused:

- Run the tests (`python -m pytest -q tests`).
- Upstream llama.cpp only: no fork patches.
- If you report a speed or quality figure, say what you measured on (GPU and RAM, llama.cpp commit, quant, context).
- Code here is MIT; by contributing you agree your contribution is MIT too.
