extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

#[proc_macro_derive(State)]
pub fn state_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => &fields.named,
            _ => panic!("State can only be derived for structs with named fields"),
        },
        _ => panic!("State can only be derived for structs"),
    };

    let update_match_arms = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().expect("Field must have a name");
        let field_name_str = field_name.to_string();
        let field_type = &f.ty;

        quote! {
            #field_name_str => {
                if let Ok(val) = value.parse::<#field_type>() {
                    self.#field_name = val;
                } else {
                    panic!("Failed to parse value '{}' for field '{}' as type {}", value, field, stringify!(#field_type));
                }
            }
        }
    });

    let get_match_arms = fields.iter().map(|f| {
        let field_name = f.ident.as_ref().expect("Field must have a name");
        let field_name_str = field_name.to_string();

        quote! {
            #field_name_str => self.#field_name.to_string()
        }
    });

    let field_name_strs_static = fields.iter().map(|f| {
        f.ident.as_ref().unwrap().to_string()
    });

    let expanded = quote! {
        impl State for #name {
            fn update_field(&mut self, field: &str, value: &str) {
                match field {
                    #(#update_match_arms,)*
                    _ => {
                        panic!("Attempted to update unknown field: {}", field);
                    }
                }
            }

            fn get_field(&self, field: &str) -> String {
                match field {
                    #(#get_match_arms,)*
                    _ => panic!("Attempted to get unknown field: {}", field)
                }
            }

            fn get_field_names() -> &'static [&'static str] {
                &[
                    #(#field_name_strs_static),*
                ]
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
            _ => panic!("StateDisplay can only be derived for structs with named fields"),
        },
        _ => panic!("StateDisplay can only be derived for structs"),
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