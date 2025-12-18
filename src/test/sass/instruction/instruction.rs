use pyo3::prelude::*;

use pyo3::types::PyTuple;

/// Fast instruction parser for any normalized SASS instructions.
///
/// Returns: (opcode, modifiers, operands, predicate)
#[pyfunction]
pub fn parse_any(py: Python<'_>, instruction: &str) -> Option<(String, Py<PyTuple>, Py<PyTuple>, Option<String>)>
{
    let mut cursor = instruction;

    // Parse an optional instruction predicate.
    let predicate: Option<String> = parse_predicate(&mut cursor);

    // Advance until non-whitespace.
    cursor = cursor.trim_start();

    // Parse opcode and modifiers
    let opcodemods_end = cursor
        .find(|c: char| c.is_whitespace())
        .unwrap_or(cursor.len());

    let opcodemods = &cursor[..opcodemods_end];

    let mut parts = opcodemods.split('.');

    let opcode: String = parts.next().unwrap().to_string();

    let modifiers = PyTuple::new(py, parts.collect::<Vec<&str>>()).unwrap().into();

    // Advance until non-whitespace.
    cursor = cursor[opcodemods_end..].trim_start();

    let operands = PyTuple::new(py, parse_operands(cursor)).unwrap().into();

    Some((opcode, modifiers, operands, predicate))
}

/// Parse an optional instruction predicate.
/// The regex would be:
///     @!?U?P(T|[0-9]+)
/// Advance the cursor past the predicate.
fn parse_predicate(cursor: &mut &str) -> Option<String>
{
    if ! cursor.starts_with('@') {
        return None;
    }

    let bytes = cursor.as_bytes();
    let mut i = 1;

    // Optional '!'.
    if i < bytes.len() && bytes[i] == b'!' { i += 1; }

    // Optional 'U'.
    if i < bytes.len() && bytes[i] == b'U' { i += 1; }

    // Required 'P'.
    if i >= bytes.len() || bytes[i] != b'P' { return None; }

    i += 1;

    // Required 'T' or digits.
    if i >= bytes.len() { return None; }

    if bytes[i] == b'T' {
        i += 1;
    } else if bytes[i].is_ascii_digit() {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
    } else {
        return None;
    }

    let predicate = cursor[..i].to_string();
    *cursor = &cursor[i..];
    Some(predicate)
}

fn parse_operands(s: &str) -> Vec<&str>
{
    if s.is_empty() { return Vec::new(); }

    if s.contains(',') {
        return s.split(',')
            .map(|op| op.trim())
            .collect();
    }
    s.split_whitespace().collect()
}
