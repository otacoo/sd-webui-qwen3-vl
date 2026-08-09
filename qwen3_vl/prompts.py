DESCRIPTION_PROMPT = (
    "Describe this image in detail."
)

SD_PROMPT = (
    "Describe this image as a Stable Diffusion prompt.\n"
    "Identify: subjects, appearance, clothing, pose, expression, "
    "environment, lighting, composition, camera perspective, artistic style.\n"
    "Return only a comma-separated list of descriptors. "
    "Do not explain your answer."
)

TAG_PROMPT = (
    "Analyze this image and generate concise visual tags.\n"
    "Return only comma-separated tags.\n"
    "Use short concrete visual descriptions.\n"
    "Describe visible characteristics only.\n"
    "Do not speculate.\n"
    "Do not write complete sentences.\n"
    "Do not provide explanations."
)

CAPTION_PROMPT = (
    "Write a concise caption describing the visual content of this image."
)

MODE_PROMPTS = {
    "Description": DESCRIPTION_PROMPT,
    "SD Prompt": SD_PROMPT,
    "Tags": TAG_PROMPT,
    "Caption": CAPTION_PROMPT,
}

MODES = list(MODE_PROMPTS.keys()) + ["Custom"]

TAG_MODES = {"Tags"}
CAPTION_MODES = {"Tags", "Caption", "SD Prompt", "Description"}