# BNF LM Code Generator

Generate code from BNF grammars using language models with step-by-step derivation.

## Usage

```bash
cargo run -- -g <grammar> -m <model> -p "<prompt>"
```

## Example

```bash
cargo run -- -g grammars/single_operation.bnf -m gemma3:4b-it-qat -p "Sum the first prime numbers" -vv
```

Output:
```
[2025-11-21T21:40:46Z INFO  bnf_graph_visualizer_2] Generated Expression: _ + _
[2025-11-21T21:40:48Z INFO  bnf_graph_visualizer_2] Generated Expression: _ + _
[2025-11-21T21:40:50Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + _
[2025-11-21T21:40:52Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + _ + _
[2025-11-21T21:40:54Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + _ + _
[2025-11-21T21:40:57Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + _
[2025-11-21T21:40:59Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + _ + _
[2025-11-21T21:41:01Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + _ + _
[2025-11-21T21:41:04Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + 5 + _
[2025-11-21T21:41:06Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + 5 + _
[2025-11-21T21:41:08Z INFO  bnf_graph_visualizer_2] Generated Expression: 2 + 3 + 5 + _
[2025-11-21T21:41:10Z INFO  bnf_graph_visualizer_2] Final Expression: 2+3+5+7
```

## Options

```
-g, --grammar-path <file>     BNF grammar file
-m, --model <name>            Ollama model name
-p, --prompt <text>           Generation task description
-v, --verbose                 Increase verbosity (-v, -vv, -vvv)
--no-log-file                 Disable file logging
```

## Grammars

Place BNF grammar files in `grammars/`:

```bnf
<program> ::= <statement>
<statement> ::= <expression> ';'
<expression> ::= <identifier> | <literal>
```

## Logs

Logs are written to `logs/bnf-visualizer-YYYYMMDD-HHMMSS.log` with full DEBUG output, while console respects verbosity level.