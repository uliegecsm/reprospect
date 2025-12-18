from reprospect._impl import tools  # type: ignore[import-not-found] # pylint: disable=no-name-in-module


def normalize(instruction: str) -> str:
    return tools.sass._decode.normalize(instruction=instruction) # pylint: disable=protected-access

def parse_instruction_and_controlcode(*,
    instruction: str,
    controlcode: str,
) -> tuple[int, str, str, str]:
    return tools.sass._decode.parse_instruction_and_controlcode( # pylint: disable=protected-access
        instruction=instruction,
        controlcode=controlcode,
    )
