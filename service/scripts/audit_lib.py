"""Shared helpers for aggregate directory and website audits.

Both audits normalize every file/URL into the same per-item dict so a single
aggregate summary can be computed and rendered consistently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from av_meta import inspect_av
from common import CONFIDENCE_LEVELS, classify_finding_confidence
from container_meta import inspect_container
from format_dispatch import classify
from image_meta import inspect_image
from score_stylometry import score_text_stylometry
from text_unicode import inspect_text


def text_hit_confidence(kind: str) -> str:
    """Layer A space homoglyphs are weaker context than invisible carriers."""
    return "informational" if kind == "space" else "probable"


def text_findings(report: Any) -> tuple[list[str], list[str], int]:
    """Flatten a TextInspectReport into finding strings + confidence lists."""
    findings: list[str] = []
    confidences: list[str] = []
    for h in report.hits:
        conf = text_hit_confidence(h.kind)
        findings.append(f"layer-a [{h.kind}] {h.label} x{h.count}")
        confidences.append(conf)
    return findings, confidences, report.suspicious_total


def scan_file(
    path: Path,
    display_name: str | None = None,
    check_stylometry: bool = False,
) -> dict[str, Any]:
    """Inspect one local file and return a normalized audit item."""
    name = display_name or str(path)
    kind = classify(path)

    if kind == "text":
        try:
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
        except OSError:
            # A file that could not be read is a FAILED SCAN, not a clean one:
            # the old swallowed-error item carried no confidence key, so
            # is_actionable answered False and every caller recorded the file
            # as scanned-and-clean (the pre-commit hook returned 0; audit_dir
            # never produced the EXIT_PARTIAL signal its wrapper exists for).
            # Let the OSError propagate: audit_dir's _scan_worker and
            # audit_website's wrapper already convert a raised exception into
            # a files_skipped entry with EXIT_PARTIAL (#158).
            raise
        report = inspect_text(text)
        findings, confidences, suspicious = text_findings(report)
        item: dict[str, Any] = {
            "path": name,
            "kind": "text",
            "has_c2pa": False,
            "has_ai_metadata": False,
            "suspicious_total": suspicious,
            "findings": findings,
            "confidence": confidences,
            "notes": report.notes,
        }
        if check_stylometry and text:
            s_rep = score_text_stylometry(text, path=name)
            item["stylometry"] = s_rep.to_dict()
            if s_rep.status == "ok" and (s_rep.score or 0.0) >= 0.65:
                item["findings"].append(
                    f"stylometry [high_probability] score {s_rep.score:.2f} ({s_rep.confidence_level})"
                )
                item["confidence"].append("probable")
                item["suspicious_total"] += 1
        return item

    if kind == "image":
        report = inspect_image(path)
        return {
            "path": name,
            "kind": report.format,
            "has_c2pa": report.has_c2pa,
            "has_ai_metadata": report.has_ai_metadata,
            "suspicious_total": 0,
            "findings": report.findings,
            "confidence": [classify_finding_confidence(f) for f in report.findings],
            "notes": report.notes,
        }

    if kind == "av":
        av_report = inspect_av(path)
        return {
            "path": name,
            "kind": av_report.format,
            "has_c2pa": av_report.has_c2pa,
            "has_ai_metadata": av_report.has_ai_metadata,
            "suspicious_total": 0,
            "findings": av_report.findings,
            "confidence": [classify_finding_confidence(f) for f in av_report.findings],
            "notes": av_report.notes,
        }

    if kind == "unknown":
        return {
            "path": name,
            "kind": "unknown",
            "has_c2pa": False,
            "has_ai_metadata": False,
            "suspicious_total": 0,
            "findings": [],
            "confidence": [],
            "notes": ["unrecognized format; not scanned"],
        }

    report = inspect_container(path)
    findings = list(report.findings)
    confidences = [classify_finding_confidence(f) for f in report.findings]
    # Layer A body-scan findings (and count) already come from
    # inspect_container() for markdown/html; it mirrors clean_container().
    suspicious = report.layer_a_total
    stylometry_dict = None

    if check_stylometry and report.format in ("html", "markdown"):
        try:
            text = path.read_text(encoding="utf-8", errors="surrogateescape")
        except OSError:
            text = ""
        if text:
            s_rep = score_text_stylometry(text, path=name)
            stylometry_dict = s_rep.to_dict()
            if s_rep.status == "ok" and (s_rep.score or 0.0) >= 0.65:
                findings.append(
                    f"stylometry [high_probability] score {s_rep.score:.2f} ({s_rep.confidence_level})"
                )
                confidences.append("probable")
                suspicious += 1

    item = {
        "path": name,
        "kind": report.format,
        "has_c2pa": report.has_c2pa,
        "has_ai_metadata": report.has_ai_metadata,
        "suspicious_total": suspicious,
        "findings": findings,
        "confidence": confidences,
        "notes": report.notes,
    }
    if stylometry_dict:
        item["stylometry"] = stylometry_dict
    return item


def is_actionable(item: dict[str, Any]) -> bool:
    """A file is actionable when it has a confirmed/probable finding or C2PA."""
    if item.get("has_c2pa"):
        return True
    return any(c in ("confirmed", "probable") for c in item.get("confidence", []))


def aggregate(files: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary block shared by directory and website audits."""
    summary = {
        "total": len(files),
        "by_kind": {},
        "with_c2pa": 0,
        "with_ai_metadata": 0,
        "with_suspicious_text": 0,
        "actionable_files": 0,
        "findings_by_confidence": {c: 0 for c in CONFIDENCE_LEVELS},
    }
    for item in files:
        kind = str(item.get("kind") or "error")
        summary["by_kind"][kind] = summary["by_kind"].get(kind, 0) + 1
        if item.get("has_c2pa"):
            summary["with_c2pa"] += 1
        if item.get("has_ai_metadata"):
            summary["with_ai_metadata"] += 1
        if item.get("suspicious_total", 0) > 0:
            summary["with_suspicious_text"] += 1
        for c in item.get("confidence", []):
            if c in summary["findings_by_confidence"]:
                summary["findings_by_confidence"][c] += 1
        if is_actionable(item):
            summary["actionable_files"] += 1
    return summary


def print_human_report(
    files: list[dict[str, Any]], summary: dict[str, Any], extra_header: dict[str, Any] | None = None
) -> None:
    """Shared plain-text rendering for audit scripts."""
    for key, value in (extra_header or {}).items():
        print(f"{key}: {value}")
    print(f"Files scanned: {summary['total']}")
    print(f"By kind: {summary['by_kind']}")
    print(f"With C2PA: {summary['with_c2pa']}")
    print(f"With AI metadata: {summary['with_ai_metadata']}")
    print(f"With suspicious text: {summary['with_suspicious_text']}")
    print(f"Actionable files: {summary['actionable_files']}")
    print(f"Findings by confidence: {summary['findings_by_confidence']}")
    for item in files:
        for msg, conf in zip(item.get("findings", []), item.get("confidence", []), strict=False):
            print(f"  [{conf}] {item['path']}: {msg}")


def format_sarif(report: dict[str, Any]) -> dict[str, Any]:
    """Convert an aggregate audit report into OASIS SARIF 2.1.0 format."""
    rules = [
        {
            "id": "AI-WATERMARK-C2PA",
            "name": "C2PAManifestDetected",
            "shortDescription": {"text": "C2PA / Content Credentials provenance manifest detected"},
            "fullDescription": {
                "text": "A C2PA provenance manifest or JUMBF metadata box was detected in the asset."
            },
            "defaultConfiguration": {"level": "error"},
            "properties": {"tags": ["provenance", "c2pa", "watermark"]},
        },
        {
            "id": "AI-WATERMARK-METADATA",
            "name": "AIMetadataMarkerDetected",
            "shortDescription": {"text": "AI generation metadata or provenance markers detected"},
            "fullDescription": {
                "text": "AI metadata markers or container generator tags were detected in the file."
            },
            "defaultConfiguration": {"level": "warning"},
            "properties": {"tags": ["provenance", "ai-generated"]},
        },
        {
            "id": "AI-WATERMARK-UNICODE-LAYER-A",
            "name": "InvisibleUnicodeWatermarkCarrier",
            "shortDescription": {
                "text": "Suspicious invisible Unicode or zero-width watermark carriers detected"
            },
            "fullDescription": {
                "text": "Invisible Unicode formatting characters or homoglyph spaces used as watermark carriers were found in the text."
            },
            "defaultConfiguration": {"level": "warning"},
            "properties": {"tags": ["watermark", "unicode", "layer-a"]},
        },
        {
            "id": "AI-STYLES-HIGH-PROBABILITY",
            "name": "HighProbabilityAITextCadence",
            "shortDescription": {
                "text": "High-probability statistical & stylometric AI text cadence detected"
            },
            "fullDescription": {
                "text": "Stylometric analysis flagged the text as highly likely to be machine-generated."
            },
            "defaultConfiguration": {"level": "note"},
            "properties": {"tags": ["stylometry", "ai-text"]},
        },
    ]

    results = []
    root_str = report.get("root", "")

    for item in report.get("files", []):
        file_path = item.get("path", "")
        if root_str:
            try:
                rel_uri = os.path.relpath(file_path, root_str).replace("\\", "/")
            except Exception:
                rel_uri = file_path.replace("\\", "/")
        else:
            rel_uri = file_path.replace("\\", "/")

        findings = item.get("findings", [])
        confidences = item.get("confidence", [])

        for msg, conf in zip(findings, confidences, strict=False):
            rule_id = "AI-WATERMARK-METADATA"
            level = "warning"

            if "c2pa" in msg.lower() or "jumbf" in msg.lower() or item.get("has_c2pa"):
                rule_id = "AI-WATERMARK-C2PA"
                level = "error"
            elif "layer-a" in msg.lower():
                rule_id = "AI-WATERMARK-UNICODE-LAYER-A"
                level = "warning" if conf in ("confirmed", "probable") else "note"
            elif "stylometry" in msg.lower():
                rule_id = "AI-STYLES-HIGH-PROBABILITY"
                level = "note"
            elif conf == "confirmed":
                level = "error"
            elif conf == "informational":
                level = "note"

            artifact_loc: dict[str, str] = {"uri": rel_uri}
            if not rel_uri.lower().startswith(("http://", "https://")):
                artifact_loc["uriBaseId"] = "%SRCROOT%"

            results.append(
                {
                    "ruleId": rule_id,
                    "level": level,
                    "message": {"text": msg},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": artifact_loc,
                            }
                        }
                    ],
                }
            )

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "watermarks-remover",
                        "version": "0.1.0",
                        "informationUri": "https://github.com/guillaumemeyer/watermarks-remover",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }
