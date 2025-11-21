use bnf::{Error, Expression, Grammar, Term};
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
You are a text generation agent guided by Backus-Naur Form (BNF) grammar rules.

Grammar:
{{Grammar}}

Task: {{Task}}

{{Context}}

Available options to continue:
{{Options}}

Instructions:
- Analyze what has been generated so far
- Consider what needs to be generated next to complete the task
- Think about which option best continues toward the goal
- Reason step-by-step about the implications of each choice
- Provide your reasoning in 2-3 sentences

Respond with ONLY your reasoning, no other text.";

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
You are a text generation agent guided by Backus-Naur Form (BNF) grammar rules.

Task: {{Task}}

{{Context}}

Your previous reasoning:
<think>
{{Reasoning}}
</think>

Which of the following options do you choose to continue?

Here are the options:
{{Options}}

Instructions:
- Based on your reasoning above, choose the best option
- You must respond **ONLY** with the number of the option you choose
- Do not add any text, periods, or explanations
- For example, if you choose option 2, respond with: 2
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

struct AgentLM {
    grammar: Grammar,
    ollama: Ollama,
    model: String,
    prompt: String,
    generation_context: std::cell::RefCell<Vec<Expression>>,
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
        let context = self.generation_context.borrow();
        let context_str = if context.is_empty() {
            String::from("You are starting the generation.")
        } else {
            let generated = self.generated_context();
            log::info!("Generated Expression: {}", generated);
            format!("So far you have generated: \"{}\"", generated)
        };

        let reasoning_prompt =
            RasoningPrompt::new(&self.grammar, &self.prompt, context_str, options.join("\n"))
                .render();

        let response = self.generation_request(&reasoning_prompt).await?;
        Ok(response.response.trim().to_string())
    }

    fn agent_prompt_with_reasoning(&self, options: &[String], reasoning: &str) -> String {
        let context = self.generation_context.borrow();

        let context_str = if context.is_empty() {
            String::from("You are starting the generation.")
        } else {
            format!(
                "So far you have generated: \"{}\"",
                self.generated_context()
            )
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
            .map(|(i, e)| format!("{}. {}", i + 1, e.to_string()))
            .collect();

        log::debug!("===== REASONING STEP =====");
        let reasoning = self.reasoning_step(&formatted_options).await?;
        log::debug!("Model reasoning: {}", reasoning);
        log::debug!("===== END REASONING =====\n");

        let result = self
            .retry_chose(&options, &formatted_options, &reasoning)
            .await;

        result.and_then(|expression| {
            self.generation_context
                .borrow_mut()
                .push((*expression).clone());
            return Ok(expression.clone());
        })
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
    #[command(flatten)]
    verbosity: clap_verbosity_flag::Verbosity,
}

fn set_verbosity(verbosity: clap_verbosity_flag::Verbosity) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_level(verbosity.log_level_filter())
        .init();
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    set_verbosity(args.verbosity);

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

        // Simulate the context based on the original structure:
        // First Expression has multiple terms: <term> "+" <expr>
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

        // Context where <expr> nonterminal is not covered
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
}
