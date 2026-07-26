from pyrag.prompts.loader import get_prompt, load_prompt_file

REQUIRED_KEYS = [
    "WikiSummaryPrompt", "WikiCandidateSlugPrompt", "WikiChunkCitationPrompt",
    "WikiPageModifyPrompt", "WikiDeduplicationPrompt",
    "WikiIndexIntroPrompt", "WikiIndexIntroUpdatePrompt", "WikiTaxonomyPlanPrompt",
]


def test_wiki_yaml_has_all_keys():
    data = load_prompt_file("wiki")
    for key in REQUIRED_KEYS:
        assert data.get(key), f"missing {key}"


def test_wiki_prompts_require_chinese_output():
    for key in ["WikiSummaryPrompt", "WikiIndexIntroPrompt", "WikiTaxonomyPlanPrompt"]:
        assert "Chinese" in get_prompt("wiki", key), key


def test_no_hardcoded_prompts_in_wiki_modules():
    import pyrag.wiki.finalize as finalize
    import pyrag.wiki.ingest as ingest
    import pyrag.wiki.merge as merge
    for mod in (ingest, merge, finalize):
        hardcoded = [n for n in vars(mod) if n.startswith("WIKI_") and n.endswith("_PROMPT")]
        assert not hardcoded, f"{mod.__name__} still hardcodes {hardcoded}"


def test_candidate_prompt_renders_granularity():
    rendered = get_prompt("wiki", "WikiCandidateSlugPrompt", {"GranularityGuidance": "<instructions>X</instructions>"})
    assert "<instructions>X</instructions>" in rendered
    assert "{{.GranularityGuidance}}" not in rendered
