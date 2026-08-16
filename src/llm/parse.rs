use serde_json::Value;

/// Pulls the first complete JSON object or array out of a model's reply.
pub fn extract_json(text: &str) -> Option<Value> {
    let body = strip_reasoning(text);
    let body = strip_fences(body);
    let candidate = balanced_slice(body)?;

    serde_json::from_str(candidate)
        .ok()
        .or_else(|| serde_json::from_str(&repair(candidate)).ok())
}

// reasoning-tuned small models emit chain of thought inline, and it is full of braces that would
// otherwise look like the answer
fn strip_reasoning(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(end) => &text[end + "</think>".len()..],
        None => text,
    }
}

// most instruct-tuned models wrap their reply in a fenced block whether or not they were asked to
fn strip_fences(text: &str) -> &str {
    let Some(open) = text.find("```") else {
        return text;
    };
    let after = &text[open + 3..];
    let body = match after.find('\n') {
        // the opening fence may carry a language tag
        Some(newline) if after[..newline].trim().chars().all(char::is_alphanumeric) => {
            &after[newline + 1..]
        }
        _ => after,
    };
    match body.find("```") {
        Some(close) => &body[..close],
        None => body,
    }
}

// ignores brackets inside string literals, so a brace in a description doesn't end the scan early
fn balanced_slice(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|b| *b == b'{' || *b == b'[')?;
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, byte) in bytes.iter().enumerate().skip(start) {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match *byte {
            b'"' => in_string = true,
            b if b == open => depth += 1,
            b if b == close => {
                depth -= 1;
                if depth == 0 {
                    // every delimiter here is ASCII, so these are char boundaries.
                    return Some(&text[start..=index]);
                }
            }
            _ => {}
        }
    }

    None
}

// the one malformation small models produce often enough to be worth handling: a trailing comma
fn repair(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut pending_comma = false;

    for ch in text.chars() {
        if in_string {
            out.push(ch);
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            ',' => {
                // hold it back until we know what follows.
                pending_comma = true;
                continue;
            }
            '}' | ']' => pending_comma = false,
            c if c.is_whitespace() => {}
            _ => {
                if pending_comma {
                    out.push(',');
                    pending_comma = false;
                }
            }
        }

        if pending_comma {
            continue;
        }

        if ch == '"' {
            in_string = true;
        }
        out.push(ch);
    }

    out
}

/// Looks a value up under any of the names a model might have used for it.
pub fn field<'a>(object: &'a Value, names: &[&str]) -> Option<&'a Value> {
    names.iter().find_map(|name| object.get(name))
}

/// Accepts a JSON number, or a number that came back quoted, which small models do freely.
pub fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn integer(value: &Value) -> Option<i64> {
    number(value).map(|n| n.round() as i64)
}

pub fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Accepts JSON booleans, the affirmative and negative words a model may use in their place, and
/// the integers one and zero.
pub fn boolean(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|n| n != 0.0),
        Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "y" | "1" => Some(true),
            "false" | "no" | "n" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// Accepts a list, or a single item where a list was asked for.
pub fn array(value: &Value) -> Vec<&Value> {
    match value {
        Value::Array(items) => items.iter().collect(),
        Value::Null => Vec::new(),
        single => vec![single],
    }
}

/// Turns a JSON schema into a filled-in example of it. Models too small to follow a schema
/// reliably will still copy the shape of an example, so backends that can't enforce a schema send
/// this instead.
pub fn sketch(schema: &Value) -> Value {
    match schema.get("type").and_then(Value::as_str) {
        Some("object") => {
            let mut object = serde_json::Map::new();
            if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
                for (name, property) in properties {
                    object.insert(name.clone(), sketch(property));
                }
            }
            Value::Object(object)
        }
        Some("array") => match schema.get("items") {
            Some(items) => Value::Array(vec![sketch(items)]),
            None => Value::Array(Vec::new()),
        },
        Some("string") => Value::String("...".to_string()),
        Some("integer") | Some("number") => Value::Number(0.into()),
        Some("boolean") => Value::Bool(false),
        _ => Value::Null,
    }
}

/// The prompt suffix for a backend that can't constrain its own output.
pub fn instructions(schema: &Value) -> String {
    format!(
        "Reply with JSON only. No prose, no explanation, no markdown fences. Match this shape \
         exactly, filling in real values:\n{}",
        serde_json::to_string_pretty(&sketch(schema)).unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_plain_json() {
        let value = extract_json(r#"{"plan": [1, 2]}"#).unwrap();
        assert_eq!(value["plan"][1], json!(2));
    }

    #[test]
    fn test_extract_from_fenced_block() {
        let reply = "Sure! Here you go:\n```json\n{\"importance\": 7}\n```\nHope that helps.";
        assert_eq!(extract_json(reply).unwrap()["importance"], json!(7));
    }

    #[test]
    fn test_extract_from_unlabelled_fence() {
        assert_eq!(extract_json("```\n{\"a\": 1}\n```").unwrap()["a"], json!(1));
    }

    #[test]
    fn test_extract_ignores_reasoning_block() {
        let reply = "<think>Maybe {\"a\": 1} is right, or maybe not</think>\n{\"a\": 2}";
        assert_eq!(extract_json(reply).unwrap()["a"], json!(2));
    }

    #[test]
    fn test_extract_ignores_surrounding_prose() {
        let reply = "The device should rest. {\"activity\": \"rest\"} That is my plan.";
        assert_eq!(extract_json(reply).unwrap()["activity"], json!("rest"));
    }

    #[test]
    fn test_extract_handles_braces_in_strings() {
        let reply = r#"{"note": "a } inside a string", "ok": true}"#;
        assert_eq!(extract_json(reply).unwrap()["ok"], json!(true));
    }

    #[test]
    fn test_extract_repairs_trailing_commas() {
        let reply = "{\"plan\": [1, 2,], \"done\": true,}";
        let value = extract_json(reply).unwrap();
        assert_eq!(value["plan"], json!([1, 2]));
        assert_eq!(value["done"], json!(true));
    }

    #[test]
    fn test_extract_top_level_array() {
        assert_eq!(extract_json("[{\"a\": 1}]").unwrap()[0]["a"], json!(1));
    }

    #[test]
    fn test_extract_rejects_unclosed_and_empty() {
        assert!(extract_json("{\"a\": 1").is_none());
        assert!(extract_json("I don't know how to answer that.").is_none());
    }

    #[test]
    fn test_lenient_scalars() {
        assert_eq!(number(&json!("7.5")), Some(7.5));
        assert_eq!(number(&json!(7)), Some(7.0));
        assert_eq!(integer(&json!("120")), Some(120));
        assert_eq!(integer(&json!(11.6)), Some(12));
        assert_eq!(boolean(&json!("yes")), Some(true));
        assert_eq!(boolean(&json!(0)), Some(false));
        assert_eq!(text(&json!(3)).as_deref(), Some("3"));
        assert_eq!(number(&json!("later")), None);
    }

    #[test]
    fn test_field_aliases() {
        let object = json!({ "duration": 30 });
        assert_eq!(
            field(&object, &["duration_minutes", "duration"]),
            Some(&json!(30))
        );
        assert!(field(&object, &["length"]).is_none());
    }

    #[test]
    fn test_array_wraps_single_values() {
        assert_eq!(array(&json!([1, 2])).len(), 2);
        assert_eq!(array(&json!("one")).len(), 1);
        assert!(array(&json!(null)).is_empty());
    }

    #[test]
    fn test_sketch_mirrors_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "react": { "type": "boolean" },
                "plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "activity": { "type": "string" },
                            "duration_minutes": { "type": "integer" }
                        }
                    }
                }
            }
        });

        assert_eq!(
            sketch(&schema),
            json!({
                "react": false,
                "plan": [{ "activity": "...", "duration_minutes": 0 }]
            })
        );
    }

    #[test]
    fn test_instructions_include_the_example() {
        let text = instructions(&json!({
            "type": "object",
            "properties": { "importance": { "type": "number" } }
        }));
        assert!(text.contains("\"importance\": 0"));
        assert!(text.contains("JSON only"));
    }
}
