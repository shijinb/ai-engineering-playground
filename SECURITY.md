# Security Policy

Please report vulnerabilities privately through GitHub's **Report a vulnerability** (Security → Advisories). You will get an acknowledgement within 7 days.

API keys are read only from environment variables by the vendor SDKs; this project never writes them to disk or logs. Telemetry records contain token counts, latency and request IDs, not prompts or outputs. Do not commit a `.env` file; `.gitignore` excludes it.

Text you send through `aiplay` goes to the provider you choose and is handled under that provider's terms.
