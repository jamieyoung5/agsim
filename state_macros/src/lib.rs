extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, parse_macro_input};

#[proc_macro_derive(State)]
pub fn state_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Error::new_spanned(input, "State requires named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return Error::new_spanned(input, "State only supports structs")
                .to_compile_error()
                .into();
        }
    };

    let diff_logic = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().expect("Field must have a name");
        let field_name_str = field_name.to_string();

        // the field name is a compile-time constant, so the event borrows it rather than owning a
        // fresh copy per change
        quote! {
            if self.#field_name != other.#field_name {
                out.push(agsim::state::StateChangeEvent {
                    time,
                    agent_id: agent_id.clone(),
                    field: ::std::borrow::Cow::Borrowed(#field_name_str),
                    old_value: agsim::state::ToValue::to_value(&self.#field_name),
                    new_value: agsim::state::ToValue::to_value(&other.#field_name),
                });
            }
        }
    });

    let expanded = quote! {
        impl agsim::state::State for #name {
            fn diff(
                &self,
                other: &Self,
                agent_id: &agsim::state::AgentId,
                time: chrono::DateTime<chrono::Utc>,
                out: &mut Vec<agsim::state::StateChangeEvent>,
            ) {
                #(#diff_logic)*
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(StateDisplay)]
pub fn state_display_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Error::new_spanned(input, "StateDisplay requires named fields")
                    .to_compile_error()
                    .into();
            }
        },
        _ => {
            return Error::new_spanned(input, "StateDisplay only supports structs")
                .to_compile_error()
                .into();
        }
    };

    let mut write_calls = Vec::new();
    let num_fields = fields.len();

    for (i, f) in fields.iter().enumerate() {
        let field_name = f.ident.as_ref().expect("Field must have a name");
        let field_name_str = field_name.to_string();

        write_calls.push(quote! {
            write!(f, "{}: {}", #field_name_str, self.#field_name)?;
        });

        if i < num_fields - 1 {
            write_calls.push(quote! {
                write!(f, " | ")?;
            });
        }
    }

    let expanded = quote! {
        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                #(#write_calls)*
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}
