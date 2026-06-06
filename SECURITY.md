# Security Policy

## Supported Versions

Resonance Graph is currently an early local-first MVP. Security fixes should target the latest `main` branch.

## Reporting A Vulnerability

If you find a vulnerability, open a private report if the GitHub repository has private vulnerability reporting enabled. If not, contact the maintainer directly before publishing details.

Please include:

- A short description of the issue.
- Steps to reproduce.
- Expected impact.
- Relevant logs or configuration details with secrets removed.

## Local Data

The app can create local media files, transcripts, embeddings, model caches, and Neo4j data. These files may contain sensitive content from processed videos.

Do not commit:

- `.env`
- `data/`
- Neo4j volumes
- Downloaded media
- Transcript JSON
- Embedding cache files
- Local model files

## Content Boundary

Do not use Resonance Graph to download or process videos unless you own them, have permission, they are Creative Commons/public-domain, or they are otherwise legally allowed to download and analyze.

The project does not support bypassing DRM, paywalls, private videos, login-only videos, region restrictions, or platform protections.
