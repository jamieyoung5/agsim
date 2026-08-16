use super::{Backend, Request};
use crate::rng;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::Tokenizer;

// ChatTemplate wraps the prompt the way the model was tuned to expect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    // Llama 3 and its derivatives
    #[default]
    Llama3,
    // ChatML: Qwen, Hermes, and most fine-tunes that aren't Llama
    ChatMl,
    // no template (the system and user text run together, for base models)
    Plain,
}

impl ChatTemplate {
    fn render(&self, system: &str, user: &str) -> String {
        match self {
            ChatTemplate::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
                 <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            ChatTemplate::ChatMl => format!(
                "<|im_start|>system\n{system}<|im_end|>\n\
                 <|im_start|>user\n{user}<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
            ChatTemplate::Plain => format!("{system}\n\n{user}\n\n"),
        }
    }

    // stop_tokens are the strings that end a turn under this template.
    fn stop_tokens(&self) -> &'static [&'static str] {
        match self {
            ChatTemplate::Llama3 => &["<|eot_id|>", "<|end_of_text|>"],
            ChatTemplate::ChatMl => &["<|im_end|>", "<|endoftext|>"],
            ChatTemplate::Plain => &["</s>"],
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Weights(candle_core::Error),
    Tokenizer(String),
    Download(String),
    Io(std::io::Error),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Weights(err) => write!(f, "loading weights: {err}"),
            LoadError::Tokenizer(err) => write!(f, "loading tokenizer: {err}"),
            LoadError::Download(err) => write!(f, "downloading from the hub: {err}"),
            LoadError::Io(err) => write!(f, "reading model files: {err}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<candle_core::Error> for LoadError {
    fn from(err: candle_core::Error) -> Self {
        LoadError::Weights(err)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(err: std::io::Error) -> Self {
        LoadError::Io(err)
    }
}

pub struct Candle {
    model: Mutex<ModelWeights>,
    tokenizer: Tokenizer,
    device: Device,
    stop_tokens: Vec<u32>,
    name: String,
    pub template: ChatTemplate,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    calls: AtomicU64,
}

impl Candle {
    pub fn from_files(
        weights: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
    ) -> Result<Self, LoadError> {
        let weights = weights.as_ref();
        let name = weights
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "candle".to_string());

        let device = default_device();
        let mut file = std::fs::File::open(weights)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        let tokenizer = Tokenizer::from_file(tokenizer.as_ref())
            .map_err(|err| LoadError::Tokenizer(err.to_string()))?;

        let mut backend = Candle {
            model: Mutex::new(model),
            tokenizer,
            device,
            stop_tokens: Vec::new(),
            name,
            template: ChatTemplate::default(),
            max_tokens: 1024,
            temperature: 0.3,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 42,
            calls: AtomicU64::new(0),
        };
        backend.resolve_stop_tokens();
        Ok(backend)
    }

    pub fn from_hub(repo: &str, file: &str, tokenizer_repo: &str) -> Result<Self, LoadError> {
        let client =
            hf_hub::HFClientSync::new().map_err(|err| LoadError::Download(err.to_string()))?;

        let (owner, name) = hf_hub::split_id(repo);
        let weights = client
            .model(owner, name)
            .download_file()
            .filename(file)
            .send()
            .map_err(|err| LoadError::Download(err.to_string()))?;

        let (owner, name) = hf_hub::split_id(tokenizer_repo);
        let tokenizer = client
            .model(owner, name)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .map_err(|err| LoadError::Download(err.to_string()))?;

        let mut backend = Self::from_files(weights, tokenizer)?;
        backend.name = repo.to_string();
        Ok(backend)
    }

    pub fn with_template(mut self, template: ChatTemplate) -> Self {
        self.template = template;
        self.resolve_stop_tokens();
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn resolve_stop_tokens(&mut self) {
        self.stop_tokens = self
            .template
            .stop_tokens()
            .iter()
            .filter_map(|token| self.tokenizer.token_to_id(token))
            .collect();
    }

    fn generate_text(&self, prompt: &str, seed: u64) -> Result<String, candle_core::Error> {
        let encoded = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        let prompt_tokens = encoded.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        let Ok(mut model) = self.model.lock() else {
            return Err(candle_core::Error::Msg("model lock poisoned".to_string()));
        };
        model.clear_kv_cache();

        let mut sampler = LogitsProcessor::from_sampling(
            seed,
            if self.temperature <= 0.0 {
                Sampling::ArgMax
            } else {
                Sampling::TopP {
                    p: self.top_p,
                    temperature: self.temperature,
                }
            },
        );

        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, 0)?.squeeze(0)?;

        let mut generated: Vec<u32> = Vec::new();
        for step in 0..self.max_tokens {
            let penalized = if self.repeat_penalty == 1.0 || generated.is_empty() {
                logits.clone()
            } else {
                let start = generated.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(&logits, self.repeat_penalty, &generated[start..])?
            };

            let next = sampler.sample(&penalized)?;
            if self.stop_tokens.contains(&next) {
                break;
            }
            generated.push(next);

            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = model
                .forward(&input, prompt_tokens.len() + step)?
                .squeeze(0)?;
        }

        self.tokenizer
            .decode(&generated, true)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))
    }
}

fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    #[cfg(feature = "metal")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    Device::Cpu
}

impl Backend for Candle {
    fn generate(&self, request: &Request) -> Option<String> {
        let prompt = self.template.render(request.system, request.user);
        let seed = rng::derive(self.seed, self.calls.fetch_add(1, Ordering::Relaxed));

        match self.generate_text(&prompt, seed) {
            Ok(text) => Some(text),
            Err(err) => {
                eprintln!("agsim::llm::candle generation failed: {err}");
                None
            }
        }
    }

    fn enforces_schema(&self) -> bool {
        false
    }

    fn label(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama3_template() {
        let prompt = ChatTemplate::Llama3.render("you are a device", "plan your day");

        assert!(prompt.starts_with("<|begin_of_text|>"));
        assert!(prompt.contains("you are a device<|eot_id|>"));
        assert!(prompt.contains("plan your day<|eot_id|>"));
        assert!(prompt.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_chatml_template() {
        let prompt = ChatTemplate::ChatMl.render("sys", "user");

        assert_eq!(
            prompt,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nuser<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_plain_template_adds_no_markers() {
        assert_eq!(ChatTemplate::Plain.render("sys", "user"), "sys\n\nuser\n\n");
    }

    #[test]
    fn test_stop_tokens_are_template_specific() {
        assert!(ChatTemplate::Llama3.stop_tokens().contains(&"<|eot_id|>"));
        assert!(ChatTemplate::ChatMl.stop_tokens().contains(&"<|im_end|>"));
    }

    #[test]
    fn test_call_seeds_differ_but_replay() {
        assert_ne!(rng::derive(7, 0), rng::derive(7, 1));
        assert_eq!(rng::derive(7, 3), rng::derive(7, 3));
    }
}
