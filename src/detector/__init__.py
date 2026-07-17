"""Standalone collapse/recovery detection, vendored from babel-ai.

These modules are extracted so the project no longer imports babel-ai as a
package. ``collapse`` is a pure function of per-round windowed similarity and has
no ML dependencies; ``recovery`` and the embedding/similarity computation are
added alongside it as they are ported and re-validated locally.
"""
