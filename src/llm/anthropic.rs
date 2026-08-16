use super::{Backend, Request};
use serde_json::{Value, json};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_MODEL: &str = "claude-opus-5";

pub struct Anthropic {
    api_key: String,
    pub model: String,
    /// Effort spent on the reasoning-heavy calls: `low`, `medium`, `high`, `xhigh`, or `max`.
    pub effort: String,
    pub max_tokens: u32,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>) -> Self {
        Anthropic {
            api_key: api_key.into(),
            model: DEFAULT_MODEL.to_string(),
            effort: "high".to_string(),
            max_tokens: 16000,
        }
    }

    pub fn from_env() -> Result<Self, std::env::VarError> {
        Ok(Self::new(std::env::var("ANTHROPIC_API_KEY")?))
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = effort.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn body(&self, request: &Request) -> Value {
        let mut output_config = json!({
            "format": { "type": "json_schema", "schema": request.schema },
        });

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": [{ "role": "user", "content": request.user }],
        });

        if request.reasoning {
            output_config["effort"] = json!(self.effort);
            body["thinking"] = json!({ "type": "adaptive" });
        } else {
            output_config["effort"] = json!("low");
            body["thinking"] = json!({ "type": "disabled" });
        }

        body["output_config"] = output_config;
        body
    }
}

fn text_of(response: &Value) -> Option<String> {
    response
        .get("content")?
        .as_array()?
        .iter()
        .find(|block| block.get("type").and_then(Value::as_str) == Some("text"))?
        .get("text")?
        .as_str()
        .map(str::to_string)
}

impl Backend for Anthropic {
    fn generate(&self, request: &Request) -> Option<String> {
        let response = match ureq::post(API_URL)
            .set("content-type", "application/json")
            .set("x-api-key", &self.api_key)
            .set("anthropic-version", "2023-06-01")
            .send_json(self.body(request))
        {
            Ok(response) => response,
            Err(err) => {
                eprintln!("agsim::llm::anthropic request failed: {err}");
                return None;
            }
        };

        text_of(&response.into_json::<Value>().ok()?)
    }

    fn enforces_schema(&self) -> bool {
        true
    }

    fn label(&self) -> String {
        self.model.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(schema: &'a Value, reasoning: bool) -> Request<'a> {
        Request {
            system: "you are a device",
            user: "plan your day",
            schema,
            reasoning,
        }
    }

    #[test]
    fn test_body_carries_the_schema() {
        let schema = json!({ "type": "object" });
        let body = Anthropic::new("key").body(&request(&schema, true));

        assert_eq!(body["model"], json!(DEFAULT_MODEL));
        assert_eq!(
            body["output_config"]["format"]["type"],
            json!("json_schema")
        );
        assert_eq!(body["output_config"]["format"]["schema"], schema);
        assert_eq!(body["messages"][0]["content"], json!("plan your day"));
    }

    #[test]
    fn test_reasoning_calls_think() {
        let schema = json!({ "type": "object" });
        let backend = Anthropic::new("key").with_effort("xhigh");

        let heavy = backend.body(&request(&schema, true));
        assert_eq!(heavy["thinking"]["type"], json!("adaptive"));
        assert_eq!(heavy["output_config"]["effort"], json!("xhigh"));

        let cheap = backend.body(&request(&schema, false));
        assert_eq!(cheap["thinking"]["type"], json!("disabled"));
        assert_eq!(cheap["output_config"]["effort"], json!("low"));
    }

    #[test]
    fn test_text_of_skips_non_text_blocks() {
        let response = json!({
            "content": [
                { "type": "thinking", "thinking": "" },
                { "type": "text", "text": "{\"plan\": []}" }
            ]
        });
        assert_eq!(text_of(&response).as_deref(), Some("{\"plan\": []}"));
        assert!(text_of(&json!({ "content": [] })).is_none());
    }
}
