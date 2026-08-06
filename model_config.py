DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini"
DEFAULT_JSON_MODEL = "gpt-4o-mini"
DEFAULT_PROSE_MODEL = "claude-sonnet-4-6"
# Used for the opt-in post-draft quality check (SOP/structural compliance + brand voice
# pass). Deliberately cheap/fast rather than the Sonnet call used for main drafting —
# swap this single constant to upgrade both passes later.
QUALITY_CHECK_MODEL = "gpt-4o-mini"
