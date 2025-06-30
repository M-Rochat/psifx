# Psifx Unit Tests

This directory contains unit tests for the psifx library.

## Purpose

Unit tests verify that individual components of the psifx library work correctly in isolation. They test specific functions, classes, and methods without relying on external dependencies.

## Structure

The unit tests are organized to mirror the structure of the psifx library:

```
unit/
├── audio/       # Tests for psifx.audio
├── io/          # Tests for psifx.io
├── text/        # Tests for psifx.text
│   └── llm/     # Tests for psifx.text.llm
├── utils/       # Tests for psifx.utils
└── video/       # Tests for psifx.video
```

## Running Unit Tests

To run only the unit tests:

```bash
pytest tests/unit/
```