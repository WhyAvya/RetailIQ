"""Audit and enrich the full Superstore Chroma metadata for C1-C3."""

from backend.chroma_metadata import audit_and_update_metadata, save_audit_report


def main() -> int:
    report = audit_and_update_metadata()
    output = save_audit_report(report)
    print(f"Saved metadata audit to {output}")
    print(f"Metadata-only chunks updated: {report['updated_chunk_count']}")
    print(f"Audit complete: {report['audit_after']['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
