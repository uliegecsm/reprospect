from reprospect._impl import tools  # type: ignore[import-not-found]


def normalize(instruction: str) -> str:
    return tools.sass._decode.normalize(instruction=instruction)

def parse_instruction_and_controlcode(*,
    instruction: str,
    controlcode: str,
) -> tuple[int, str, str, str]:
    return tools.sass._decode.parse_instruction_and_controlcode(
        instruction=instruction,
        controlcode=controlcode,
    )
