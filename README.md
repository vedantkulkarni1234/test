# Multi-CVE Reconnaissance Script

**A production-ready Bash script for non-destructive detection of critical Java and Node.js vulnerabilities.**

## ⚠️ Disclaimer

This tool is **for authorized security testing only**. It performs non-destructive reconnaissance and does **NOT**:
- Include exploit code or destructive payloads
- Create files or persistent changes on target systems
- Guarantee the absence of vulnerabilities (only attempts detection)
- Require explicit authorization from target system owners

**Always obtain proper written authorization before scanning any system.**

## Features

This script detects 9 critical vulnerabilities with high confidence:

| CVE ID | Vulnerability | Component | Version Range |
|--------|---------------|-----------|----------------|
| CVE-2022-22965 | Spring4Shell | Spring Framework | 5.2.0-5.2.19, 5.3.0-5.3.17 |
| CVE-2021-44228 | Log4Shell | Apache Log4j | 2.0.0-2.14.1+ |
| CVE-2022-42889 | Text4Shell | Apache Commons Text | 1.5-1.9 |
| CVE-2017-18349 | Fastjson RCE | Alibaba Fastjson | <1.2.42 |
| CVE-2019-12384 | Jackson RCE | Jackson | <2.9.8.1, <2.8.11.2 |
| CVE-2017-5638 | Struts2 RCE | Apache Struts 2 | 2.3.5-2.5.10 |
| CVE-2019-7609 | Kibana PPCE | Elastic Kibana | 5.0.0-5.6.13, 6.0.0-6.5.3 |
| CVE-2019-11358 | Ghostscript RCE | Ghostscript/ImageMagick | Depends on integration |
| CVE-2023-37466 | vm2 Sandbox Escape | Node.js vm2 | <3.9.10 |

### Detection Confidence Levels

For each vulnerability, the script reports:

- **`confirmed_vulnerable`**: Very high confidence based on version strings, explicit headers, or known endpoints
- **`likely_vulnerable`**: Strong evidence (e.g., vulnerable library detected, vulnerable endpoint found)
- **`potentially_vulnerable`**: Weak/indirect indicators (e.g., general framework detected)
- **`not_detected`**: No evidence found (but doesn't prove safety)

### Detection Strategy

The script uses **deterministic, low-false-positive methods**:

1. **Version fingerprinting**: Checks headers, error pages, and version endpoints for known vulnerable versions
2. **Endpoint discovery**: Looks for framework-specific paths (e.g., `/actuator` for Spring, `/struts2-showcase` for Struts2)
3. **Library detection**: Identifies presence of vulnerable libraries in responses and headers
4. **Configuration detection**: Checks for dangerous configurations (e.g., JNDI endpoints, polymorphic deserialization)
5. **Benign test inputs**: Submits minimal, non-destructive payloads to test processing behavior

## Installation

```bash
# Clone and prepare
git clone <repo>
cd multi-cve-recon

# Make executable
chmod +x multi-cve-recon.sh

# (Optional) Copy to system path
sudo cp multi-cve-recon.sh /usr/local/bin/
```

## Usage

### Basic Syntax

```bash
./multi-cve-recon.sh [-f targets.txt] [-t target_url] [-k] [-o output.jsonl] [-h]
```

### Options

| Option | Description |
|--------|-------------|
| `-f FILE` | Read targets from file (one target per line) |
| `-t TARGET` | Scan a single target (URL with scheme: `http://...` or `https://...`) |
| `-k` | Ignore TLS/SSL verification (useful for self-signed certificates) |
| `-o OUTPUT` | Write results to file in JSONL format (default: stdout) |
| `-h, --help` | Show help message |

### Examples

#### Scan a single target
```bash
./multi-cve-recon.sh -t http://target.example.com:8080
```

#### Scan multiple targets from file
```bash
./multi-cve-recon.sh -f targets.txt
```

#### Scan with self-signed TLS
```bash
./multi-cve-recon.sh -t https://target.local:8443 -k
```

#### Save results to file
```bash
./multi-cve-recon.sh -f targets.txt -o results.jsonl
```

#### Scan with all options
```bash
./multi-cve-recon.sh -f targets.txt -k -o results.jsonl
```

### Target File Format

Create `targets.txt` with one target per line:

```
http://192.168.1.100:8080
https://app.internal.company:8443
http://legacy-server.local
https://api.example.com
```

Comments and empty lines are automatically skipped:

```
# Internal targets
http://internal-app:8080

# Production
https://api.prod.company.com

# Staging
http://stage-app:3000
```

## Output Format

Results are output as **JSON Lines** (one JSON object per result):

```json
{"target":"http://example.com","cve_id":"CVE-2022-22965","name":"Spring4Shell","status":"likely_vulnerable","evidence":"Spring actuator endpoint accessible"}
{"target":"http://example.com","cve_id":"CVE-2021-44228","name":"Log4Shell","status":"potentially_vulnerable","evidence":"Java application detected"}
{"target":"http://example.com","cve_id":"CVE-2017-5638","name":"Apache Struts 2 RCE","status":"not_detected","evidence":""}
```

### Parsing Results

**With `jq`:**

```bash
# Extract confirmed vulnerabilities
cat results.jsonl | jq 'select(.status=="confirmed_vulnerable")'

# Count vulnerabilities by type
cat results.jsonl | jq -r '.cve_id' | sort | uniq -c

# Filter results by CVE
cat results.jsonl | jq 'select(.cve_id=="CVE-2022-22965")'
```

**With `grep` and `awk`:**

```bash
# Show only confirmed or likely vulnerabilities
cat results.jsonl | grep -E '"status":"(confirmed|likely)_vulnerable'

# Extract targets with Spring4Shell vulnerability
cat results.jsonl | grep "CVE-2022-22965" | grep "vulnerable"
```

## Technical Implementation

### Architecture

The script is organized into:

1. **Configuration section**: Timeout, headers, colors, arrays
2. **Utility functions**: curl wrapping, URL normalization, JSON parsing, output logging
3. **Detection functions**: One per vulnerability family
4. **Main orchestration**: Target loading, scanning loop, summary reporting

### Key Design Decisions

#### Minimal External Dependencies
- Uses only `bash`, `curl`, `grep`, `sed`, `awk`
- Avoids `jq` (optional for parsing results, not used in detection)
- Compatible with standard Linux distros (CentOS, Ubuntu, Debian, Alpine)

#### Low False Positives
- **No regex pattern matching** on generic strings
- **Deterministic checks**: Exact version ranges, explicit library names, specific endpoints
- **Context-aware**: Combines multiple signals (e.g., framework + endpoint + version)
- **Explicit library detection**: Looks for library names in headers and responses, not guessing

#### Non-Destructive
- **No payload execution**: Only submits benign test strings
- **No file writes**: Doesn't create or modify files on targets
- **No resource exhaustion**: Uses standard timeouts (10 seconds per request)
- **Safe on production**: Can be run on authorized production systems

#### Parallel-Safe (with modifications)
The script processes targets sequentially by default. For large scans, modify with `&` for background jobs:

```bash
for target in "${targets_list[@]}"; do
    scan_target "$target" &
done
wait
```

### Function Reference

#### `check_spring4shell()`
**Detects**: Spring Framework versions 5.2.0-5.2.19, 5.3.0-5.3.17
- Checks for `/actuator/env` endpoint (Spring Boot indicator)
- Looks for Spring version in responses and headers
- Tests health endpoints for Spring presence

**Confidence**: Medium (strong if actuator found, weak if just "Spring" string)

#### `check_log4shell()`
**Detects**: Apache Log4j 2.0.0-2.14.1
- Identifies Java applications
- Checks for logging endpoints
- Looks for Log4j version strings in error pages

**Confidence**: High if version found, medium if logging endpoints detected

#### `check_text4shell()`
**Detects**: Apache Commons Text 1.5-1.9
- Identifies applications using Apache Commons libraries
- Tests text processing endpoints
- Looks for version indicators

**Confidence**: Low to medium (library presence is weak indicator alone)

#### `check_fastjson_rce()`
**Detects**: Alibaba Fastjson < 1.2.42
- Tests JSON API endpoints
- Checks for Fastjson headers or version strings
- Looks for autoType patterns in responses

**Confidence**: High if Fastjson explicitly detected, medium if JSON endpoint found

#### `check_jackson_rce()`
**Detects**: Jackson < 2.9.8.1, < 2.8.11.2
- Tests JSON endpoints with polymorphic type hints
- Checks for Jackson in headers or responses
- Looks for `@class` patterns in responses

**Confidence**: High if Jackson detected, medium if polymorphic JSON found

#### `check_struts2_rce()`
**Detects**: Apache Struts 2 versions 2.3.5-2.5.10
- Checks for Struts2 showcase or example paths
- Tests for OGNL expression endpoints (`.action` files)
- Looks for Struts-specific headers

**Confidence**: High if version found, medium if action endpoints detected

#### `check_kibana_rce()`
**Detects**: Kibana 5.0.0-5.6.13, 6.0.0-6.5.3
- Tests Kibana-specific endpoints (`/app/kibana`, `/api/status`)
- Checks for Kibana strings in responses
- Extracts version from status API

**Confidence**: High if version in vulnerable range found, medium if Kibana detected

#### `check_ghostscript_rce()`
**Detects**: Ghostscript/ImageMagick integration vulnerabilities
- Looks for file upload endpoints
- Checks for ImageMagick or Ghostscript headers
- Identifies image/PDF processing capabilities

**Confidence**: Low to medium (integration-dependent)

#### `check_vm2_rce()`
**Detects**: Node.js vm2 < 3.9.10
- Identifies Node.js applications
- Tests for code execution/sandbox endpoints
- Checks `package.json` for vm2 version
- Looks for vulnerable version ranges

**Confidence**: High if package.json found, medium if sandbox endpoint detected

## Limitations and Caveats

1. **Fingerprinting dependent**: Detection relies on discoverable headers, endpoints, and responses. Hardened systems may not reveal version info.

2. **No runtime analysis**: Cannot detect vulnerabilities without version indicators or specific endpoints.

3. **Time-dependent**: Vulnerability status is accurate as of this script's creation. New patched versions may be released.

4. **Framework obfuscation**: Systems that obscure or hide framework information will show lower confidence.

5. **False negatives possible**: A system may be vulnerable but not detected if:
   - Endpoints are not exposed
   - Version info is hidden
   - Non-standard paths are used
   - Security headers mask application details

6. **Timeout-dependent**: Slow or unreliable networks may cause timeouts. Adjust `CURL_TIMEOUT` if needed.

## Extending the Script

### Add a New CVE Check

```bash
check_new_cve() {
    local target="$1"
    local status="not_detected"
    local evidence=""
    
    # Your detection logic here
    # Make curl requests, parse responses
    
    # Report result
    output_result "$target" "CVE-XXXX-XXXXX" "New CVE Name" "$status" "$evidence"
}
```

Then add to the `scan_target()` function:
```bash
check_new_cve "$target"
```

### Modify Timeout

Edit the script and change:
```bash
CURL_TIMEOUT=10  # Change this value
```

### Change Output Format

Modify the `output_result()` function to output CSV, XML, or other formats.

### Add Proxy Support

Modify `curl_request()` to add:
```bash
if [[ -n "$HTTP_PROXY" ]]; then
    curl_opts+=("--proxy" "$HTTP_PROXY")
fi
```

## Performance Considerations

- **Parallel scanning**: The script processes targets sequentially. For large batches, run multiple instances or modify for background jobs.
- **Timeout setting**: Default 10-second timeout per request. Adjust for slow/unreliable networks.
- **Request overhead**: ~10 requests per target (one per CVE + variant checks). ~100 requests per target is typical.

### Estimated Execution Time

- Single target: 30-120 seconds (depending on target responsiveness)
- 10 targets: 5-20 minutes
- 100 targets: 50-200 minutes

To speed up, reduce `CURL_TIMEOUT` or implement parallel scanning.

## Security Considerations

### Safe to run on production?

**Yes, if authorized.** The script:
- Makes only read-only HTTP requests
- Doesn't submit exploit payloads
- Doesn't create or delete files
- Doesn't establish persistent connections
- Times out after 10 seconds per request

However, always:
1. Obtain written authorization
2. Run during maintenance windows if possible
3. Monitor target logs for scanning activity
4. Coordinate with target system owners

### Data Privacy

The script:
- Stores results only where you direct (file or stdout)
- Doesn't send data to external servers
- Doesn't require authentication
- Can be used offline by modifying target list

## Troubleshooting

### Script hangs on a target

- Target may be slow or unresponsive
- Ctrl+C to interrupt, skip with Ctrl+C
- Increase `CURL_TIMEOUT` if network is slow

### High number of false positives

- Check individual evidence fields to validate findings
- Spring Framework alone is not conclusive
- Java + Log4j + logging endpoint is stronger evidence
- Always manually verify high-confidence results

### Targets not scanned

- Check URL format: must include `http://` or `https://`
- Verify file exists and is readable: `cat targets.txt`
- Check for empty lines or invisible characters
- Run with `-t` single target to test one URL

### No results produced

- Verify targets are reachable: `curl -I http://target:port`
- Check TLS issues: Add `-k` flag for self-signed certs
- Confirm targets respond to HTTP requests
- Increase timeout if network is slow

## License

This script is provided as-is for authorized security testing. Use responsibly and ethically.

## Contributing

To improve detection accuracy:
1. Test against known vulnerable systems
2. Report false positives with target details (anonymized)
3. Suggest additional detection methods for each CVE
4. Share improvements via pull requests

## References

- [CVE-2022-22965 - Spring4Shell](https://nvd.nist.gov/vuln/detail/CVE-2022-22965)
- [CVE-2021-44228 - Log4Shell](https://nvd.nist.gov/vuln/detail/CVE-2021-44228)
- [CVE-2022-42889 - Text4Shell](https://nvd.nist.gov/vuln/detail/CVE-2022-42889)
- [CVE-2017-18349 - Fastjson RCE](https://nvd.nist.gov/vuln/detail/CVE-2017-18349)
- [CVE-2019-12384 - Jackson RCE](https://nvd.nist.gov/vuln/detail/CVE-2019-12384)
- [CVE-2017-5638 - Struts2 RCE](https://nvd.nist.gov/vuln/detail/CVE-2017-5638)
- [CVE-2019-7609 - Kibana Prototype Pollution](https://nvd.nist.gov/vuln/detail/CVE-2019-7609)
- [CVE-2023-37466 - vm2 Sandbox Escape](https://nvd.nist.gov/vuln/detail/CVE-2023-37466)
