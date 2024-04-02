# LLMLean

#### LLM on your laptop:
1. Install [ollama](https://ollama.com/).

2. Pull a language model:
```bash
ollama pull solobsd/llemma-7b
```

4. Add `llmlean` to lakefile:
```lean
require llmlean from git
  "https://github.com/cmu-l3/llmlean.git"
```

3. Import:
```lean
import LLMlean
```
Now use a tactic described below.

#### LLM in the cloud:

1. Get a [together.ai](https://www.together.ai/) API key.

2. Set environment variables in VS Code. Example:

<img src="img/env_example1.png" style="width:500px">

Now use a tactic described below.

----
### `llmstep` tactic
Next-tactic suggestions via `llmstep "{prefix}"`. Examples:

- `llmstep ""`

  <img src="img/llmstep_empty.png" style="width:500px">

- `llmstep "apply "`

  <img src="img/llmstep_apply.png" style="width:500px">

---------------

## Customization

Please see the following:
1. [Customization](docs/customization.md)
