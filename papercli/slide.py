"""Slide generation from article text highlights."""

from typing import TYPE_CHECKING, Callable, Literal

from papercli.models import ArticleHighlights

if TYPE_CHECKING:
    from papercli.cache import Cache
    from papercli.gemini import GeminiClient

# Supported slide styles
SlideStyle = Literal["handdrawn", "minimal", "academic", "dark", "colorful"]

VALID_STYLES: set[str] = {"handdrawn", "minimal", "academic", "dark", "colorful"}

# Style descriptions for image generation prompts
STYLE_PROMPTS: dict[str, str] = {
    "handdrawn": (
        "Hand-drawn sketch style with marker pen strokes, imperfect lines, "
        "doodle-like illustrations, notebook paper texture, casual handwritten fonts, "
        "warm earthy colors (brown, orange, cream), slight paper grain effect."
    ),
    "minimal": (
        "Ultra-minimalist clean design with lots of white space, thin geometric lines, "
        "monochromatic color scheme (black, white, single accent color), "
        "modern sans-serif typography, subtle shadows, no decorative elements."
    ),
    "academic": (
        "Professional academic poster style with structured grid layout, "
        "serif typography, muted blue and gray color palette, subtle gradients, "
        "clean data visualization elements, institutional feel."
    ),
    "dark": (
        "Dark futuristic tech theme with deep navy/black background, "
        "neon accent colors (cyan, magenta, electric blue), glowing effects, "
        "monospace or tech-style fonts, circuit board patterns, high contrast."
    ),
    "colorful": (
        "Vibrant and energetic design with bold saturated colors, "
        "playful geometric shapes, gradient backgrounds, modern rounded elements, "
        "dynamic composition with visual hierarchy, friendly and engaging feel."
    ),
}

# System instruction for highlight extraction
HIGHLIGHT_EXTRACTION_SYSTEM = """You are an expert at extracting key highlights from academic articles and research papers.

Your task is to analyze the provided text and extract the most important points for a visual slide summary.

Output a JSON object with the following structure:
{
  "title": "A concise, engaging title (max 10 words)",
  "subtitle": "Optional context or subtitle (null if not needed)",
  "bullets": ["Point 1", "Point 2", ...],  // 3-6 key highlights, each max 15 words
  "takeaway": "One sentence main conclusion or insight",
  "keywords": ["keyword1", "keyword2", ...]  // 3-5 visual theme keywords
}

Guidelines:
- Title should capture the main topic/contribution
- Bullets should be concrete findings, methods, or insights (not vague statements)
- Each bullet should stand alone and be understandable without context
- Takeaway should be actionable or memorable
- Keywords help with visual styling (e.g., "biology", "AI", "climate", "medicine")

Return ONLY the JSON object, no markdown or explanation."""


def _build_highlight_prompt(text: str, num_bullets: int) -> str:
    """Build prompt for highlight extraction."""
    return f"""Analyze this article text and extract {num_bullets} key highlights.

ARTICLE TEXT:
---
{text[:12000]}
---

Extract the highlights as specified. Return JSON only."""


def _build_slide_prompt(
    highlights: ArticleHighlights,
    style: SlideStyle,
    aspect_ratio: str,
) -> str:
    """Build prompt for slide image generation."""
    style_desc = STYLE_PROMPTS.get(style, STYLE_PROMPTS["minimal"])

    # Format bullets as numbered list
    bullets_text = "\n".join(f"  {i+1}. {b}" for i, b in enumerate(highlights.bullets))

    # Include keywords for visual context
    keywords_hint = ""
    if highlights.keywords:
        keywords_hint = f"\nVisual theme keywords: {', '.join(highlights.keywords)}"

    return f"""Create a single-page presentation slide with the following content:

TITLE: {highlights.title}
{f'SUBTITLE: {highlights.subtitle}' if highlights.subtitle else ''}

KEY POINTS:
{bullets_text}

TAKEAWAY: {highlights.takeaway}
{keywords_hint}

DESIGN REQUIREMENTS:
- Style: {style_desc}
- Aspect ratio: {aspect_ratio} (landscape orientation)
- Single page, no multi-slide layout
- Clear visual hierarchy: title > bullets > takeaway
- All text must be readable and not cut off
- Professional presentation quality
- Include subtle visual elements related to the topic
- Balanced composition with appropriate margins

Generate this as a complete, polished presentation slide image."""


async def summarize_highlights(
    text: str,
    client: "GeminiClient",
    cache: "Cache | None" = None,
    num_bullets: int = 5,
    progress_callback: Callable[[str], None] | None = None,
) -> ArticleHighlights:
    """
    Extract key highlights from article text using Gemini.

    Args:
        text: Article text content
        client: GeminiClient instance
        cache: Optional cache for results
        num_bullets: Target number of bullet points (1-8)
        progress_callback: Optional callback for progress updates

    Returns:
        ArticleHighlights with extracted content

    Raises:
        GeminiError: If API call fails
        ValueError: If response cannot be parsed
    """
    from papercli.cache import Cache

    # Clamp bullets to valid range
    num_bullets = max(1, min(8, num_bullets))

    # Check cache
    if cache:
        cache_key = Cache.hash_key("slide_highlights", text[:2000], str(num_bullets))
        cached = await cache.get(cache_key)
        if cached:
            if progress_callback:
                progress_callback("Using cached highlights")
            return ArticleHighlights.model_validate(cached)

    if progress_callback:
        progress_callback("Extracting highlights from text...")

    prompt = _build_highlight_prompt(text, num_bullets)

    response = await client.generate_text_json(
        prompt=prompt,
        system_instruction=HIGHLIGHT_EXTRACTION_SYSTEM,
        temperature=0.7,
        max_tokens=2000,
    )

    # Validate response structure
    highlights = ArticleHighlights.model_validate(response)

    # Cache result
    if cache:
        await cache.set(cache_key, highlights.model_dump())

    return highlights


async def render_slide_image(
    highlights: ArticleHighlights,
    style: SlideStyle,
    client: "GeminiClient",
    cache: "Cache | None" = None,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    progress_callback: Callable[[str], None] | None = None,
) -> bytes:
    """
    Generate a slide image from highlights using Gemini.

    Args:
        highlights: Extracted article highlights
        style: Visual style preset
        client: GeminiClient instance
        cache: Optional cache for results
        aspect_ratio: Image aspect ratio (16:9, 4:3, 1:1)
        image_size: Image size (1K, 2K, 4K)
        progress_callback: Optional callback for progress updates

    Returns:
        PNG image bytes

    Raises:
        GeminiError: If image generation fails
    """
    from papercli.cache import Cache

    # Validate style
    if style not in VALID_STYLES:
        style = "minimal"

    # Check cache (only for smaller images to avoid sqlite bloat)
    cache_key = None
    if cache and image_size in ("1K", "2K"):
        cache_key = Cache.hash_key(
            "slide_image",
            highlights.model_dump_json(),
            style,
            aspect_ratio,
            image_size,
        )
        cached = await cache.get(cache_key)
        if cached and isinstance(cached, str):
            if progress_callback:
                progress_callback("Using cached slide image")
            import base64
            return base64.b64decode(cached)

    if progress_callback:
        progress_callback(f"Generating {style} slide image...")

    prompt = _build_slide_prompt(highlights, style, aspect_ratio)

    image_bytes, mime_type = await client.generate_image(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        temperature=0.8,
    )

    # Cache result (base64 encoded, only for smaller images)
    if cache and cache_key and len(image_bytes) < 2_000_000:  # < 2MB
        import base64
        await cache.set(cache_key, base64.b64encode(image_bytes).decode("ascii"))

    return image_bytes


async def generate_slide(
    text: str,
    style: SlideStyle,
    client: "GeminiClient",
    cache: "Cache | None" = None,
    num_bullets: int = 5,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[bytes, ArticleHighlights]:
    """
    Full pipeline: extract highlights and generate slide image.

    Args:
        text: Article text content
        style: Visual style preset
        client: GeminiClient instance
        cache: Optional cache
        num_bullets: Target number of bullet points
        aspect_ratio: Image aspect ratio
        image_size: Image size
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (image_bytes, highlights)
    """
    # Step 1: Extract highlights
    highlights = await summarize_highlights(
        text=text,
        client=client,
        cache=cache,
        num_bullets=num_bullets,
        progress_callback=progress_callback,
    )

    # Step 2: Generate slide image
    image_bytes = await render_slide_image(
        highlights=highlights,
        style=style,
        client=client,
        cache=cache,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        progress_callback=progress_callback,
    )

    return image_bytes, highlights

