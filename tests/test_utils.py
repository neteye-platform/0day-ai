import pytest

from utils.general import (
    deduplicate_cves,
    dispatch_claim_taken,
    get_canonical_id,
    is_path_excluded,
    parse_call_target,
    safe_cache_filename,
    take_dispatch_claim,
    take_feedback_dispatch,
)
from utils.stats import strip_step_numbering


class TestParseCallTarget:
    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            ("Class::method", ("Class", "method")),
            ("module.method", ("module", "method")),
            ("plain_function", ("", "plain_function")),
            ("", ("", "")),
            ("Renderer::getInstance()->display", ("Renderer", "display")),
            ("$db->query", ("", "query")),
            ("$this->save", ("$this", "save")),
            ("obj.method(arg1, arg2)", ("obj", "method")),
            ("a.b.c", ("a.b", "c")),
        ],
    )
    def test_splits_module_and_symbol(self, target, expected):
        assert parse_call_target(target) == expected

    def test_none_is_tolerated(self):
        assert parse_call_target(None) == ("", "")


class TestDispatchClaims:
    def test_first_claim_wins(self):
        assert take_dispatch_claim("v1", "x", 0)
        assert not take_dispatch_claim("v1", "x", 0)

    def test_distinct_keys_are_independent(self):
        assert take_dispatch_claim("v1", "x", 0)
        assert take_dispatch_claim("v1", "x", 1)
        assert take_dispatch_claim("v2", "x", 0)

    def test_peek_does_not_consume(self):
        assert not dispatch_claim_taken("v1")
        assert not dispatch_claim_taken("v1")
        assert take_dispatch_claim("v1")
        assert dispatch_claim_taken("v1")

    def test_feedback_claim_is_per_round(self):
        assert take_feedback_dispatch("v1", 0)
        assert not take_feedback_dispatch("v1", 0)
        assert take_feedback_dispatch("v1", 1)


class TestStripStepNumbering:
    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ("1. POST /login", "POST /login"),
            ("2) GET /x", "GET /x"),
            ("  10.  spaced", "spaced"),
            ("no numbering", "no numbering"),
            ("v1.2 is not a step", "v1.2 is not a step"),
        ],
    )
    def test_removes_only_leading_numbers(self, raw, clean):
        assert strip_step_numbering(raw) == clean


class TestIsPathExcluded:
    @pytest.mark.parametrize(
        "path",
        [
            "app/vendor/lib/x.php",
            "web/node_modules/pkg/index.js",
            "src/.git/config",
            "README.md",
            "notes.txt",
            "report_2026-01-31_235959/findings/poc.py",
        ],
    )
    def test_noise_is_excluded(self, path):
        assert is_path_excluded(path)

    @pytest.mark.parametrize(
        "path",
        [
            "src/app/controller.py",
            "index.php",
            # only the exact timestamped report directory is blinded
            "report_generator/main.py",
        ],
    )
    def test_first_party_sources_are_kept(self, path):
        assert not is_path_excluded(path)

    def test_empty_path_is_not_excluded(self):
        assert not is_path_excluded("")


class TestSafeCacheFilename:
    def test_path_separators_are_flattened(self):
        assert "/" not in safe_cache_filename("dependency:golang.org/x/text.json")
        assert "\\" not in safe_cache_filename("a\\b.json")

    def test_short_names_are_untouched(self):
        assert safe_cache_filename("abc.json") == "abc.json"

    def test_long_names_are_truncated_deterministically(self):
        name = "n" * 400 + ".json"
        out = safe_cache_filename(name)
        assert len(out.encode()) <= 240
        assert out.endswith(".json")
        assert out == safe_cache_filename(name)
        assert out != safe_cache_filename("m" * 400 + ".json")

    def test_multibyte_names_stay_within_the_byte_limit(self):
        assert len(safe_cache_filename("é" * 300 + ".json").encode()) <= 240


class TestCanonicalId:
    def test_alias_wins(self):
        assert get_canonical_id({"id": "GHSA-x", "aliases": ["CVE-2024-1"]}) == (
            "CVE-2024-1"
        )

    def test_embedded_id(self):
        assert get_canonical_id({"id": "DEBIAN-CVE-2023-99"}) == "CVE-2023-99"

    def test_falls_back_to_the_record_id(self):
        assert get_canonical_id({"id": "GHSA-abcd"}) == "GHSA-abcd"


class TestDeduplicateCves:
    def advisory(self, vid, details, **kw):
        return {
            "id": vid,
            "aliases": ["CVE-2024-1"],
            "details": details,
            "affected": [{"package": {"name": "lib"}}],
            **kw,
        }

    def test_advisories_of_one_cve_collapse(self):
        out = deduplicate_cves(
            [
                self.advisory("GHSA-1", "Short."),
                self.advisory("PYSEC-1", "A much longer description of the flaw."),
            ]
        )
        assert len(out) == 1
        assert out[0]["id"] == "CVE-2024-1"
        assert out[0]["package"] == "lib"
        assert out[0]["details"] == "A much longer description of the flaw."

    def test_contained_descriptions_are_collapsed(self):
        out = deduplicate_cves(
            [
                self.advisory("A", "Remote code execution"),
                self.advisory("B", "remote  code execution in the parser."),
            ]
        )
        assert len(out[0]["descriptions"]) == 1

    def test_enrichment_is_backfilled_from_any_advisory(self):
        out = deduplicate_cves(
            [
                self.advisory("A", "A long enough description text."),
                self.advisory(
                    "B",
                    "short",
                    database_specific={"cwe_ids": ["CWE-79"], "severity": "high"},
                    severity=[{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N"}],
                    affected=[
                        {
                            "package": {"name": "lib"},
                            "ranges": [{"events": [{"fixed": "1.2.3"}]}],
                        }
                    ],
                ),
            ]
        )[0]
        assert out["cwe_ids"] == ["CWE-79"]
        assert out["severity_label"] == "HIGH"
        assert out["fixed_version"] == "1.2.3"
        assert out["cvss_vector"] == "CVSS:3.1/AV:N"

    def test_records_without_details_are_dropped(self):
        assert deduplicate_cves([self.advisory("A", "")]) == []

    def test_distinct_cves_stay_separate(self):
        out = deduplicate_cves(
            [
                self.advisory("A", "one"),
                {**self.advisory("B", "two"), "aliases": ["CVE-2024-2"]},
            ]
        )
        assert {r["id"] for r in out} == {"CVE-2024-1", "CVE-2024-2"}
