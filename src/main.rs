use bnf::{Error, Expression, Grammar, Term};
use chrono::Local;
use clap::Parser;
use log;
use ollama_rs::generation::completion::GenerationResponse;
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest};
use regex::Regex;
use std::fs;
use std::future::Future;
use std::pin::Pin;

trait Template {
    fn render(&self) -> String;
}

struct RasoningPrompt<'a> {
    grammar: &'a Grammar,
    task: &'a String,
    context: String,
    options: String,
}

impl<'a> RasoningPrompt<'a> {
    fn new(grammar: &'a Grammar, task: &'a String, context: String, options: String) -> Self {
        RasoningPrompt {
            grammar,
            task,
            context,
            options,
        }
    }
}

const RASONING_PROMPT_TEMPLATE: &str = "
You are performing a step-by-step BNF grammar derivation to generate valid code.

Goal: {{Task}}

Grammar (relevant rules):
{{Grammar}}

{{Context}}

Available options to continue:
{{Options}}

Instructions:
1. Look carefully at what has been generated so far
2. Understand the current structure and what nonterminals need expansion
3. Consider which option will correctly build toward the goal
4. Each option shows what will be produced when chosen
5. Think about the STRUCTURE you're building, not just individual tokens

Provide your reasoning in 2-3 sentences explaining:
- What you need to generate next
- Why the chosen option moves toward the goal
- How it fits into the overall structure

Reasoning:";

impl<'a> Template for RasoningPrompt<'a> {
    fn render(&self) -> String {
        RASONING_PROMPT_TEMPLATE
            .replace("{{Grammar}}", &self.grammar.to_string())
            .replace("{{Task}}", &self.task)
            .replace("{{Context}}", &self.context)
            .replace("{{Options}}", &self.options)
    }
}

struct AgentPrompt<'a> {
    task: &'a String,
    context: String,
    reasoning: String,
    options: String,
}

impl<'a> AgentPrompt<'a> {
    fn new(task: &'a String, context: String, reasoning: String, options: String) -> Self {
        AgentPrompt {
            task,
            context,
            reasoning,
            options,
        }
    }
}

const AGENT_PROMPT_TEMPLATE: &str = "
You are generating code to: {{Task}}

Your reasoning:
<think>{{Reasoning}}</think>

{{Context}}

Available options:
{{Options}}

Based on your reasoning above, choose the option number that correctly continues the derivation.

CRITICAL: Respond with ONLY the option number (e.g., 2)
Do NOT add any explanation, punctuation, or other text.
";

impl<'a> Template for AgentPrompt<'a> {
    fn render(&self) -> String {
        AGENT_PROMPT_TEMPLATE
            .replace("{{Task}}", &self.task)
            .replace("{{Context}}", &self.context)
            .replace("{{Reasoning}}", &self.reasoning)
            .replace("{{Options}}", &self.options)
    }
}

trait Agent {
    async fn generate(&self) -> Result<String, Error>;
}

struct Generator<A: Agent> {
    agent: A,
}

impl<A: Agent> Generator<A> {
    async fn generate(&self) -> Result<String, Error> {
        self.agent.generate().await
    }
}

#[derive(Clone)]
struct DecisionHistory {
    rule: String,
    choice: String,
    result: String,
}

struct AgentLM {
    grammar: Grammar,
    ollama: Ollama,
    model: String,
    prompt: String,
    generation_context: std::cell::RefCell<Vec<Expression>>,
    decision_history: std::cell::RefCell<Vec<DecisionHistory>>,
}

impl AgentLM {
    fn new(grammar: Grammar, model: String, prompt: String) -> Self {
        let ollama = Ollama::default();
        AgentLM {
            grammar,
            ollama,
            model,
            prompt,
            generation_context: std::cell::RefCell::new(Vec::new()),
            decision_history: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn generated_context(&self) -> String {
        let context = self.generation_context.borrow().clone();

        if context.is_empty() {
            return String::new();
        }

        let (collapsed, _) = self.collapse_expressions(&context, 0);

        collapsed.join(" ")
    }

    fn partial_derivation(&self) -> String {
        let context = self.generation_context.borrow();

        if context.is_empty() {
            return String::new();
        }

        let first_expr = &context[0];
        let terms: Vec<String> = first_expr
            .terms_iter()
            .map(|term| match term {
                Term::Terminal(t) => format!("'{}'", t),
                Term::Nonterminal(nt) => nt.clone(),
                Term::AnonymousNonterminal(_) => String::from("<anon>"),
            })
            .collect();

        terms.join(" ")
    }

    fn format_decision_history(&self) -> String {
        let history = self.decision_history.borrow();

        if history.is_empty() {
            return String::new();
        }

        let formatted = history
            .iter()
            .enumerate()
            .map(|(i, decision)| {
                format!(
                    "  {}. {} → '{}' → {}",
                    i + 1,
                    decision.rule,
                    decision.choice,
                    decision.result
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!("\n\nDecision history:\n{}", formatted)
    }

    fn generated_context_display(&self) -> String {
        let collapsed = self.generated_context();
        let partial = self.partial_derivation();

        if collapsed.is_empty() {
            if partial.is_empty() {
                String::from("You are starting the generation.")
            } else {
                format!("Current structure: {}", partial)
            }
        } else {
            if partial.is_empty() {
                format!("So far you have generated: \"{}\"", collapsed)
            } else {
                format!(
                    "So far you have generated: \"{}\"\nCurrent expansion: {}",
                    collapsed, partial
                )
            }
        }
    }

    fn collapse_expressions(
        &self,
        context: &[Expression],
        start_index: usize,
    ) -> (Vec<String>, usize) {
        if start_index >= context.len() {
            return (Vec::new(), start_index);
        }

        let head = &context[start_index];

        head.terms_iter().fold(
            (Vec::new(), start_index + 1),
            |(mut acc, index), term| match term {
                Term::Terminal(t) => {
                    acc.push(t.clone());
                    (acc, index)
                }
                Term::Nonterminal(_) => {
                    let (expanded, new_index) = self.collapse_expressions(context, index);
                    if expanded.is_empty() {
                        acc.push(String::from("_"));
                        (acc, index)
                    } else {
                        acc.extend(expanded);
                        (acc, new_index)
                    }
                }
                Term::AnonymousNonterminal(_) => (acc, index),
            },
        )
    }

    async fn reasoning_step(&self, options: &[String]) -> Result<String, Error> {
        let generated = self.generated_context();
        log::info!("Generated Expression: {}", generated);

        let context_display = self.generated_context_display();
        let history = self.format_decision_history();
        let context_str = format!("{}{}", context_display, history);

        let reasoning_prompt =
            RasoningPrompt::new(&self.grammar, &self.prompt, context_str, options.join("\n"))
                .render();

        let response = self.generation_request(&reasoning_prompt).await?;
        Ok(response.response.trim().to_string())
    }

    fn agent_prompt_with_reasoning(&self, options: &[String], reasoning: &str) -> String {
        let context = self.generation_context.borrow();
        let context_str = if context.is_empty() {
            String::from("")
        } else {
            self.generated_context()
        };

        AgentPrompt::new(
            &self.prompt,
            context_str,
            String::from(reasoning),
            options.join("\n"),
        )
        .render()
    }

    async fn generation_request(&self, prompt: &str) -> Result<GenerationResponse, Error> {
        let request = GenerationRequest::new(self.model.clone(), prompt);
        self.ollama
            .generate(request)
            .await
            .map_err(|e| Error::GenerateError(format!("Ollama error: {}", e)))
    }

    async fn choose(&self, options: &Vec<&Expression>) -> Result<Expression, Error> {
        if options.len() == 1 {
            return Ok(options[0].clone());
        }

        let formatted_options: Vec<String> = options
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let terms_display: Vec<String> = e
                    .terms_iter()
                    .map(|term| match term {
                        Term::Terminal(t) => format!("'{}'", t),
                        Term::Nonterminal(nt) => nt.clone(),
                        Term::AnonymousNonterminal(_) => String::from("<anon>"),
                    })
                    .collect();
                format!(
                    "{}. {} → produces: {}",
                    i + 1,
                    e.to_string(),
                    terms_display.join(" ")
                )
            })
            .collect();

        log::debug!("===== REASONING STEP =====");
        let reasoning = self
            .reasoning_step(&formatted_options)
            .await
            .unwrap_or(String::new());
        log::debug!("{}", reasoning);
        log::debug!("===== END REASONING =====\n");

        let result = self
            .retry_chose(&options, &formatted_options, &reasoning)
            .await;

        result.and_then(|expression| {
            let expr_clone = (*expression).clone();
            self.generation_context
                .borrow_mut()
                .push(expr_clone.clone());

            let rule = self.get_current_nonterminal();
            let choice = expr_clone.to_string();
            let result_str = self.generated_context();

            self.decision_history.borrow_mut().push(DecisionHistory {
                rule,
                choice,
                result: result_str,
            });

            return Ok(expr_clone);
        })
    }

    fn get_current_nonterminal(&self) -> String {
        let context = self.generation_context.borrow();
        if context.is_empty() {
            return String::from("<start>");
        }

        if let Some(last_expr) = context.last() {
            for term in last_expr.terms_iter() {
                if let Term::Nonterminal(nt) = term {
                    return nt.clone();
                }
            }
        }

        String::from("<unknown>")
    }

    fn parse_index(response: &str) -> Result<usize, Error> {
        let re = Regex::new(r"\d+").unwrap();
        re.find(response)
            .and_then(|m| m.as_str().parse::<usize>().ok())
            .ok_or_else(|| {
                Error::ParseError(format!("No valid number found in response: '{}'", response))
            })
    }

    fn parse_choose_from_generation<'a>(
        &self,
        result: Result<GenerationResponse, Error>,
        options: &'a Vec<&'a Expression>,
    ) -> Result<&'a Expression, Error> {
        let parsed_result = result.and_then(|r| {
            log::debug!("LM response: '{}'", r.response.trim());
            r.response
                .trim()
                .parse::<usize>()
                .or_else(|_| Self::parse_index(r.response.trim()))
                .or_else(|error| {
                    Err(Error::ParseError(format!(
                        "Invalid digit '{}'. Error: {}",
                        r.response,
                        error.to_string()
                    )))
                })
        });

        let choose_expression = parsed_result.and_then(|index| {
            let array_index = index.saturating_sub(1);
            log::debug!("Parsed index: {}, array index: {}", index, array_index);
            options.get(array_index).ok_or(Error::ParseError(format!(
                "Invalid response: index {} is out of bounds. Length: {}",
                index,
                options.len()
            )))
        });

        choose_expression.and_then(|choose| {
            log::debug!("Chosen expression: {:?}", choose);
            Ok(*choose)
        })
    }

    async fn retry_chose<'a>(
        &self,
        options: &'a Vec<&'a Expression>,
        formatted_options: &[String],
        reasoning: &str,
    ) -> Result<&'a Expression, Error> {
        let prompt = self.agent_prompt_with_reasoning(&formatted_options, &reasoning);
        log::debug!("===== FULL PROMPT SENT TO LM =====");
        log::debug!("{}", prompt);
        log::debug!("===== END PROMPT =====");

        let mut last_error = None;
        for attempt in 1..=3 {
            if attempt > 1 {
                log::warn!("Retry attempt {}", attempt);
            }

            let generation_result = self.generation_request(&prompt).await;
            match self.parse_choose_from_generation(generation_result, &options) {
                Ok(term) => return Ok(term),
                Err(e) => last_error = Some(e),
            }
        }

        match last_error {
            Some(e) => Err(e),
            None => Err(Error::ParseError("No valid response found".to_string())),
        }
    }

    fn traverse<'a>(
        &'a self,
        ident: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + 'a>> {
        Box::pin(async move {
            let nonterm = Term::Nonterminal(ident.to_string());
            let find_lhs = self.grammar.productions_iter().find(|&x| x.lhs == nonterm);

            let production = match find_lhs {
                Some(p) => p,
                None => return Ok(nonterm.to_string()),
            };

            let expressions = production.rhs_iter().collect::<Vec<&Expression>>();

            let expression = match self.choose(&expressions).await {
                Ok(e) => e,
                Err(e) => {
                    return Err(e);
                }
            };

            let mut result = String::new();
            for term in expression.terms_iter() {
                let term_result = self.eval_terminal(term).await?;
                result.push_str(&term_result);
            }

            Ok(result)
        })
    }

    fn eval_terminal<'a>(
        &'a self,
        term: &'a Term,
    ) -> Pin<Box<dyn Future<Output = Result<String, Error>> + 'a>> {
        Box::pin(async move {
            match term {
                Term::Nonterminal(nt) => self.traverse(nt).await,
                Term::Terminal(t) => Ok(t.clone()),
                Term::AnonymousNonterminal(rhs) => {
                    let rhs_refs: Vec<&Expression> = rhs.iter().collect();
                    let expression = match self.choose(&rhs_refs).await {
                        Ok(e) => e,
                        Err(e) => return Err(e),
                    };

                    let mut result = String::new();
                    for term in expression.terms_iter() {
                        let term_result = self.eval_terminal(term).await?;
                        result.push_str(&term_result);
                    }

                    Ok(result)
                }
            }
        })
    }
}

impl Agent for AgentLM {
    async fn generate(&self) -> Result<String, Error> {
        self.generation_context.borrow_mut().clear();
        self.decision_history.borrow_mut().clear();

        let start_rule: String;
        let first_production = self.grammar.productions_iter().next();

        if !self.grammar.terminates() {
            return Err(Error::GenerateError(
                "Can't generate, first rule in grammar doesn't lead to a terminal state"
                    .to_string(),
            ));
        }

        match first_production {
            Some(term) => match term.lhs {
                Term::Nonterminal(ref nt) => start_rule = nt.clone(),
                Term::Terminal(_) => {
                    return Err(Error::GenerateError(format!(
                        "Terminal type cannot define a production in '{term}'!"
                    )));
                }
                Term::AnonymousNonterminal(_) => {
                    return Err(Error::GenerateError(String::from(
                        "Anonymous nonterminals are not allowed",
                    )));
                }
            },
            None => {
                return Err(Error::GenerateError(String::from(
                    "Failed to get first production!",
                )));
            }
        }
        let result = self.traverse(&start_rule).await;

        return result;
    }
}

impl Generator<AgentLM> {
    fn new(grammar: Grammar, model: String, prompt: String) -> Self {
        Self {
            agent: AgentLM::new(grammar.clone(), model, prompt),
        }
    }
}

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    grammar_path: std::path::PathBuf,
    #[arg(short, long, default_value = "gemma3:4b-it-qat")]
    model: String,
    #[arg(short, long)]
    prompt: String,
    #[arg(long, help = "Disable logging to file (only log to console)")]
    no_log_file: bool,
    #[command(flatten)]
    verbosity: clap_verbosity_flag::Verbosity,
}

fn setup_logging(
    verbosity: clap_verbosity_flag::Verbosity,
    enable_file_logging: bool,
) -> Result<(), fern::InitError> {
    let console_level = verbosity.log_level_filter();

    let format =
        |out: fern::FormatCallback, message: &std::fmt::Arguments, record: &log::Record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.target(),
                message
            ))
        };

    let console_dispatch = fern::Dispatch::new()
        .format(format)
        .level(console_level)
        .chain(std::io::stdout());

    if enable_file_logging {
        std::fs::create_dir_all("logs").ok();

        let log_file = format!(
            "logs/bnf-visualizer-{}.log",
            Local::now().format("%Y%m%d-%H%M%S")
        );

        let file_dispatch = fern::Dispatch::new()
            .format(format)
            .level(log::LevelFilter::Debug)
            .chain(fern::log_file(&log_file)?);

        fern::Dispatch::new()
            .chain(console_dispatch)
            .chain(file_dispatch)
            .apply()?;

        log::info!("Logging to console and file: {}", log_file);
    } else {
        console_dispatch.apply()?;
        log::info!("Logging to console only");
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    setup_logging(args.verbosity, !args.no_log_file).expect("Failed to initialize logging");

    let input = fs::read_to_string(args.grammar_path).expect("Failed to read grammar file");

    match input.parse() {
        Ok(grammar) => {
            let generated = Generator::<AgentLM>::new(grammar, args.model, args.prompt)
                .generate()
                .await;

            match generated {
                Ok(result) => log::info!("Final Expression: {}", result),
                Err(error) => log::error!("Something went wrong: {}!", error),
            }
        }
        Err(error) => log::error!("Error parsing grammar: {}", error),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collapse_expressions() {
        let grammar = "
            <expr> ::= <term> | <term> '+' <expr>
            <term> ::= <number>
            <number> ::= '1' | '2'
        "
        .parse::<Grammar>()
        .unwrap();

        let agent = AgentLM::new(
            grammar,
            String::from("test-model"),
            String::from("test task"),
        );

        let context = vec![
            Expression::from_parts(vec![
                Term::Nonterminal(String::from("<term>")),
                Term::Terminal(String::from("+")),
                Term::Nonterminal(String::from("<expr>")),
            ]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<number>"))]),
            Expression::from_parts(vec![Term::Terminal(String::from("1"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<term>"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<number>"))]),
            Expression::from_parts(vec![Term::Terminal(String::from("2"))]),
        ];

        let (collapsed, final_index) = agent.collapse_expressions(&context, 0);

        assert_eq!(collapsed, vec!["1", "+", "2"]);
        assert_eq!(final_index, 6);
    }

    #[test]
    fn test_collapse_expressions_with_uncovered_nonterminal() {
        let grammar = "
            <expr> ::= <term> | <term> '+' <expr>
            <term> ::= <number>
            <number> ::= '1' | '2'
        "
        .parse::<Grammar>()
        .unwrap();

        let agent = AgentLM::new(
            grammar,
            String::from("test-model"),
            String::from("test task"),
        );

        let context = vec![
            Expression::from_parts(vec![
                Term::Nonterminal(String::from("<term>")),
                Term::Terminal(String::from("+")),
                Term::Nonterminal(String::from("<expr>")),
            ]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<number>"))]),
            Expression::from_parts(vec![Term::Terminal(String::from("1"))]),
        ];

        let (collapsed, final_index) = agent.collapse_expressions(&context, 0);

        assert_eq!(collapsed, vec!["1", "+", "_"]);
        assert_eq!(final_index, 3);
    }

    #[test]
    fn test_collapse_javascript_console_log() {
        let grammar_str = std::fs::read_to_string("grammars/javascript.bnf")
            .expect("Failed to read javascript.bnf");

        let grammar = grammar_str.parse::<Grammar>().unwrap();

        let agent = AgentLM::new(
            grammar,
            String::from("test-model"),
            String::from("test task"),
        );

        let context = vec![
            Expression::from_parts(vec![
                Term::Nonterminal(String::from("<member_expression>")),
                Term::Terminal(String::from("(")),
                Term::Nonterminal(String::from("<argument_list>")),
                Term::Terminal(String::from(")")),
                Term::Terminal(String::from(";")),
            ]),
            Expression::from_parts(vec![
                Term::Nonterminal(String::from("<identifier>")),
                Term::Terminal(String::from(".")),
                Term::Nonterminal(String::from("<identifier>")),
            ]),
            Expression::from_parts(vec![Term::Terminal(String::from("console"))]),
            Expression::from_parts(vec![Term::Terminal(String::from("log"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<expression>"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<primary>"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<literal>"))]),
            Expression::from_parts(vec![Term::Nonterminal(String::from("<string>"))]),
            Expression::from_parts(vec![
                Term::Terminal(String::from("\"")),
                Term::Nonterminal(String::from("<string_content>")),
                Term::Terminal(String::from("\"")),
            ]),
            Expression::from_parts(vec![Term::Terminal(String::from("Hello World"))]),
        ];

        let (collapsed, final_index) = agent.collapse_expressions(&context, 0);

        assert_eq!(
            collapsed,
            vec![
                "console",
                ".",
                "log",
                "(",
                "\"",
                "Hello World",
                "\"",
                ")",
                ";"
            ]
        );
        assert_eq!(final_index, 10);
    }

    #[test]
    fn test_decision_history_and_partial_derivation() {
        let grammar = "
            <expr> ::= <term> | <term> '+' <expr>
            <term> ::= <number>
            <number> ::= '1' | '2'
        "
        .parse::<Grammar>()
        .unwrap();

        let agent = AgentLM::new(
            grammar,
            String::from("test-model"),
            String::from("test task"),
        );

        agent
            .generation_context
            .borrow_mut()
            .push(Expression::from_parts(vec![
                Term::Nonterminal(String::from("<term>")),
                Term::Terminal(String::from("+")),
                Term::Nonterminal(String::from("<expr>")),
            ]));

        let partial = agent.partial_derivation();
        assert_eq!(partial, "<term> '+' <expr>");

        agent
            .generation_context
            .borrow_mut()
            .push(Expression::from_parts(vec![Term::Nonterminal(
                String::from("<number>"),
            )]));
        agent
            .generation_context
            .borrow_mut()
            .push(Expression::from_parts(vec![Term::Terminal(String::from(
                "1",
            ))]));

        let generated = agent.generated_context();
        assert_eq!(generated, "1 + _");

        agent.decision_history.borrow_mut().push(DecisionHistory {
            rule: String::from("<expr>"),
            choice: String::from("<term> '+' <expr>"),
            result: String::from("1 + _"),
        });

        let history = agent.format_decision_history();
        assert!(history.contains("Decision history:"));
        assert!(history.contains("<expr>"));
        assert!(history.contains("<term> '+' <expr>"));

        let display = agent.generated_context_display();
        assert!(display.contains("So far you have generated"));
        assert!(display.contains("1 + _"));
    }
}
