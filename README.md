# BNF_LM_code_generator

An experiment to generate code from a BNF grammar guide using an LM.

## Run

```
$ cargo run -- -g grammars/single_operation.bnf -m gemma3:4b-it-qat -p "Sum all  prime numbers"
```

### Example

```
cargo run -- -g grammars/single_operation.bnf -m gemma3:4b-it-qat -p "Sum the first prime numbers" -vv
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
