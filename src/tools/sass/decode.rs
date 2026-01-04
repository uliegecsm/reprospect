use std::borrow::Cow;

#[pyo3::pyfunction]
#[inline]
pub fn normalize(instruction: &str) -> Cow<'_, str>
{
    let bytes = instruction.as_bytes();
    let len = bytes.len();

    // First pass.
    // Count occurrences of each pattern.
    let mut bracket_space_count = 0usize;
    let mut space_comma_count = 0usize;
    let mut i = 0;

    while i + 1 < len
    {
        if bytes[i] == b']' && bytes[i + 1] == b' ' && i + 2 < len && bytes[i + 2] == b'['
        {
            bracket_space_count += 1;
            i += 3;
        }
        else if bytes[i] == b' ' && bytes[i + 1] == b','
        {
            space_comma_count += 1;
            i += 2;
        }
        else { i += 1; }
    }

    // No replacements needed.
    if bracket_space_count == 0 && space_comma_count == 0 {
        return Cow::Borrowed(instruction);
    }

    // Second pass.
    // Build the result with exact capacity.
    let new_len = len - bracket_space_count - space_comma_count;
    let mut result = Vec::with_capacity(new_len);
    let mut i = 0;

    while i < len
    {
        if i + 2 < len && bytes[i] == b']' && bytes[i + 1] == b' ' && bytes[i + 2] == b'['
        {
            result.extend_from_slice(b"][");
            i += 3;
        }
        else if i + 1 < len &&  bytes[i] == b' ' && bytes[i + 1] == b','
        {
            result.push(b',');
            i += 2;
        }
        else
        {
            result.push(bytes[i]);
            i += 1;
        }
    }

    // Only ASCII bytes were removed, other bytes were copied unchanged.
    // No need to pay any safety check here.
    Cow::Owned(unsafe{String::from_utf8_unchecked(result)})
}

#[pyo3::pyfunction]
pub fn parse_instruction_and_controlcode(instruction: &str, controlcode: &str) -> pyo3::PyResult<(u32, String, String, String)>
{
    // Parse offset, assuming it is formatted as:
    //      /* <offset> */
    let instruction_offset_start = instruction.find("/*").unwrap() + 2;
    let instruction_offset_end = instruction[instruction_offset_start..].find("*/").unwrap() + instruction_offset_start;

    let instruction_offset = u32::from_str_radix(&instruction[instruction_offset_start..instruction_offset_end], 16).unwrap();

    // Parse instruction hex, assuming it is formatted as:
    //      /* <hex> */)
    let instruction_hex_start = instruction.rfind("/*").unwrap() + 3;
    let instruction_hex_end = instruction[instruction_hex_start..].find("*/").unwrap() + instruction_hex_start - 1;

    let instruction_hex = instruction[instruction_hex_start..instruction_hex_end].to_string();

    // Parse the instruction, assuming it is in-between the offset and hex.
    // Normalize it.
    let instruction_slice = &instruction[instruction_offset_end + 2..instruction_hex_start - 3];
    let instruction_normalized = normalize(
        instruction_slice
        .split(|c| c == ';' || c == '?' || c == '&')
        .next()
        .unwrap()
        .trim()
    ).to_string();

    // Parse the control code.
    let controlcode_start = controlcode.rfind("/*").unwrap() + 2;
    let controlcode_end = controlcode[controlcode_start..].find("*/").unwrap() + controlcode_start;
    let controlcode_hex = controlcode[controlcode_start..controlcode_end].trim().to_string();

    Ok((instruction_offset, instruction_normalized, instruction_hex, controlcode_hex))
}
