use super::{Backend, Request};
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchemaMode {
    #[default]
    Prompted,
    JsonObject,
    JsonSchema,
}

pub struct OpenAiCompat {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f64,
    pub schema_mode: SchemaMode,
    pub attempts: u32,
}

impl OpenAiCompat {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        OpenAiCompat {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            agent: ureq::AgentBuilder::new()
                .timeout_read(Duration::from_secs(300))
                .build(),
            model: model.into(),
            max_tokens: 2048,
            temperature: 0.2,
            schema_mode: SchemaMode::default(),
            attempts: 3,
        }
    }

    pub fn from_env() -> Result<Self, std::env::VarError> {
        let backend = Self::new(
            std::env::var("AGSIM_LLM_URL")?,
            std::env::var("AGSIM_LLM_MODEL")?,
        );
        Ok(match std::env::var("AGSIM_LLM_KEY") {
            Ok(key) => backend.with_api_key(key),
            Err(_) => backend,
        })
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_schema_mode(mut self, mode: SchemaMode) -> Self {
        self.schema_mode = mode;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.agent = ureq::AgentBuilder::new().timeout_read(timeout).build();
        self
    }

    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts;
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn body(&self, request: &Request) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                { "role": "system", "content": request.system },
                { "role": "user", "content": request.user },
            ],
        });

        match self.schema_mode {
            SchemaMode::Prompted => {}
            SchemaMode::JsonObject => {
                body["response_format"] = json!({ "type": "json_object" });
            }
            SchemaMode::JsonSchema => {
                body["response_format"] = json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agsim_response",
                        "strict": true,
                        "schema": request.schema,
                    },
                });
            }
        }

        body
    }
}

fn content_of(response: &Value) -> Option<String> {
    response
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

impl Backend for OpenAiCompat {
    fn generate(&self, request: &Request) -> Option<String> {
        let mut call = self
            .agent
            .post(&self.endpoint())
            .set("content-type", "application/json");
        if let Some(key) = &self.api_key {
            call = call.set("authorization", &format!("Bearer {key}"));
        }

        let response = match call.send_json(self.body(request)) {
            Ok(response) => response,
            Err(err) => {
                eprintln!(
                    "agsim::llm::openai request to {} failed: {err}",
                    self.endpoint()
                );
                return None;
            }
        };

        content_of(&response.into_json::<Value>().ok()?)
    }

    fn enforces_schema(&self) -> bool {
        self.schema_mode == SchemaMode::JsonSchema
    }

    fn attempts(&self) -> u32 {
        if self.enforces_schema() {
            1
        } else {
            self.attempts
        }
    }

    fn label(&self) -> String {
        self.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(schema: &'a Value) -> Request<'a> {
        Request {
            system: "you are a device",
            user: "plan your day",
            schema,
            reasoning: true,
        }
    }

    #[test]
    fn test_endpoint_normalizes_the_base_url() {
        let with_slash = OpenAiCompat::new("http://localhost:11434/v1/", "qwen3");
        let without = OpenAiCompat::new("http://localhost:11434/v1", "qwen3");

        assert_eq!(with_slash.endpoint(), without.endpoint());
        assert_eq!(
            without.endpoint(),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[test]
    fn test_prompted_mode_sends_no_response_format() {
        let schema = json!({ "type": "object" });
        let body = OpenAiCompat::new("http://localhost:8080/v1", "small").body(&request(&schema));

        assert!(body.get("response_format").is_none());
        assert_eq!(body["messages"][0]["role"], json!("system"));
        assert_eq!(body["messages"][1]["content"], json!("plan your day"));
    }

    #[test]
    fn test_json_schema_mode_constrains_the_reply() {
        let schema = json!({ "type": "object", "properties": {} });
        let backend = OpenAiCompat::new("http://localhost:8080/v1", "small")
            .with_schema_mode(SchemaMode::JsonSchema);
        let body = backend.body(&request(&schema));

        assert_eq!(body["response_format"]["type"], json!("json_schema"));
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert!(backend.enforces_schema());
        assert_eq!(backend.attempts(), 1);
    }

    #[test]
    fn test_json_object_mode_is_still_coaxed() {
        let backend = OpenAiCompat::new("http://localhost:8080/v1", "small")
            .with_schema_mode(SchemaMode::JsonObject)
            .with_attempts(2);
        let body = backend.body(&request(&json!({ "type": "object" })));

        assert_eq!(body["response_format"], json!({ "type": "json_object" }));
        assert!(!backend.enforces_schema());
        assert_eq!(backend.attempts(), 2);
    }

    #[test]
    fn test_content_of() {
        let response = json!({ "choices": [{ "message": { "content": "{\"a\": 1}" } }] });
        assert_eq!(content_of(&response).as_deref(), Some("{\"a\": 1}"));
        assert!(content_of(&json!({ "choices": [] })).is_none());
    }
}
