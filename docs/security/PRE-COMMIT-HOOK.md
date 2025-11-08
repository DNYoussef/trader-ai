# Pre-Commit Hook for Secret Protection

This document provides instructions for setting up a pre-commit hook to prevent accidental commits of sensitive files.

---

## Quick Setup

### Linux/Mac

```bash
# Navigate to trader-ai repository
cd /path/to/trader-ai

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
# Pre-commit hook to prevent committing sensitive files

echo "🔍 Checking for sensitive files..."

# Check for .env files
if git diff --cached --name-only | grep -E '\.env$|\.env\.'; then
    echo "❌ ERROR: .env file detected in commit"
    echo ""
    echo "  Attempted to commit sensitive environment file."
    echo "  This could expose API keys and secrets."
    echo ""
    echo "  To remove from staging:"
    echo "    git reset HEAD .env"
    echo ""
    exit 1
fi

# Check for common secret file patterns
if git diff --cached --name-only | grep -E 'secrets?\.(json|yml|yaml|toml)|credentials\.(json|yml|yaml|toml)'; then
    echo "❌ ERROR: Potential secrets file detected"
    echo ""
    echo "  Files matching secrets/credentials pattern found."
    echo "  Review the file before committing."
    echo ""
    exit 1
fi

# Check for private keys
if git diff --cached --name-only | grep -E '\.(key|pem|p12|pfx)$'; then
    echo "❌ ERROR: Private key file detected"
    echo ""
    echo "  Private key files should never be committed."
    echo "  Use environment variables or secret management instead."
    echo ""
    exit 1
fi

# Simple pattern matching for API keys in file contents
# (Only check staged files to avoid false positives)
for file in $(git diff --cached --name-only | grep -E '\.(py|js|ts|jsx|tsx|json|yml|yaml)$'); do
    if [ -f "$file" ]; then
        # Check for common API key patterns
        if grep -q -E '(ALPACA_API_KEY|PLAID_SECRET|JWT_SECRET_KEY|HF_TOKEN|DATABASE_ENCRYPTION_KEY)\s*=\s*["\x27][a-zA-Z0-9_\-]{20,}' "$file"; then
            echo "⚠️  WARNING: Potential hardcoded secret in $file"
            echo ""
            echo "  Detected pattern matching API key format."
            echo "  Please verify this is not a real secret."
            echo ""
            echo "  If this is a false positive, you can bypass with:"
            echo "    git commit --no-verify"
            echo ""
            read -p "  Continue anyway? (yes/no): " choice
            case "$choice" in
                yes|y|Y) ;;
                *) exit 1;;
            esac
        fi
    fi
done

echo "✅ Pre-commit checks passed"
exit 0
EOF

# Make hook executable
chmod +x .git/hooks/pre-commit

echo "✅ Pre-commit hook installed successfully"
```

### Windows (Git Bash)

```bash
# Navigate to trader-ai repository
cd /c/Users/YourName/Desktop/trader-ai

# Create pre-commit hook (same content as above)
# Copy the script from the Linux/Mac section above

# Make executable (Git Bash)
chmod +x .git/hooks/pre-commit
```

### Windows (PowerShell)

```powershell
# Navigate to trader-ai repository
cd C:\Users\YourName\Desktop\trader-ai

# Create pre-commit hook
$hookContent = @'
#!/bin/sh
# Pre-commit hook to prevent committing sensitive files

echo "🔍 Checking for sensitive files..."

# Check for .env files
if git diff --cached --name-only | grep -E '\.env$|\.env\.'; then
    echo "❌ ERROR: .env file detected in commit"
    echo ""
    echo "  Attempted to commit sensitive environment file."
    echo "  This could expose API keys and secrets."
    echo ""
    echo "  To remove from staging:"
    echo "    git reset HEAD .env"
    echo ""
    exit 1
fi

# Check for common secret file patterns
if git diff --cached --name-only | grep -E 'secrets?\.(json|yml|yaml|toml)|credentials\.(json|yml|yaml|toml)'; then
    echo "❌ ERROR: Potential secrets file detected"
    echo ""
    echo "  Files matching secrets/credentials pattern found."
    echo "  Review the file before committing."
    echo ""
    exit 1
fi

# Check for private keys
if git diff --cached --name-only | grep -E '\.(key|pem|p12|pfx)$'; then
    echo "❌ ERROR: Private key file detected"
    echo ""
    echo "  Private key files should never be committed."
    echo "  Use environment variables or secret management instead."
    echo ""
    exit 1
fi

echo "✅ Pre-commit checks passed"
exit 0
'@

# Write hook file
$hookContent | Out-File -FilePath .git\hooks\pre-commit -Encoding ASCII -NoNewline

Write-Host "✅ Pre-commit hook installed successfully"
```

---

## Testing the Hook

### Test 1: Try to commit .env file

```bash
# This should FAIL
git add .env
git commit -m "Test commit"

# Expected output:
# ❌ ERROR: .env file detected in commit
```

### Test 2: Commit safe files

```bash
# This should SUCCEED
git add README.md
git commit -m "Update README"

# Expected output:
# ✅ Pre-commit checks passed
```

### Test 3: Bypass hook (emergency only)

```bash
# Use --no-verify to bypass (use sparingly!)
git commit --no-verify -m "Emergency commit"
```

---

## What the Hook Checks

1. **Environment Files**
   - `.env`
   - `.env.local`
   - `.env.development`
   - `.env.production`

2. **Secrets Files**
   - `secrets.json`
   - `credentials.yml`
   - `api_keys.toml`

3. **Private Keys**
   - `*.key`
   - `*.pem`
   - `*.p12`
   - `*.pfx`

4. **Hardcoded Secrets** (Pattern matching in files)
   - Detects patterns like: `ALPACA_API_KEY="pk_live_..."`
   - Warns on potential hardcoded credentials
   - Allows bypass with confirmation

---

## Customization

### Add More File Patterns

Edit `.git/hooks/pre-commit` and add to the grep pattern:

```bash
# Add custom patterns
if git diff --cached --name-only | grep -E 'config\.prod\.json|production\.config'; then
    echo "❌ ERROR: Production config file detected"
    exit 1
fi
```

### Add More Secret Patterns

Add additional regex patterns for secret detection:

```bash
# Check for AWS keys
if grep -q -E 'AKIA[0-9A-Z]{16}' "$file"; then
    echo "⚠️  WARNING: Potential AWS access key in $file"
fi

# Check for private SSH keys
if grep -q -E '-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----' "$file"; then
    echo "⚠️  WARNING: SSH private key detected in $file"
fi
```

---

## Troubleshooting

### Hook Not Running

1. **Check file permissions**:
   ```bash
   ls -l .git/hooks/pre-commit
   # Should show: -rwxr-xr-x (executable)
   ```

2. **Make executable**:
   ```bash
   chmod +x .git/hooks/pre-commit
   ```

3. **Check git config**:
   ```bash
   git config --get core.hooksPath
   # Should be empty or point to .git/hooks
   ```

### False Positives

If the hook blocks legitimate commits:

1. **Review the warning** - Is there actually a secret?
2. **Fix the code** - Remove any hardcoded values
3. **Bypass if necessary** - Use `--no-verify` (document why)

### Hook Doesn't Detect Secrets

1. **Test pattern matching**:
   ```bash
   echo 'PLAID_SECRET="test123"' | grep -E 'PLAID_SECRET\s*=\s*["\x27][a-zA-Z0-9_\-]{20,}'
   ```

2. **Check file encoding** - Hook expects UTF-8
3. **Update patterns** - Add more specific regex

---

## Alternative: Using Pre-Commit Framework

For a more robust solution, use the [pre-commit](https://pre-commit.com/) framework:

### 1. Install pre-commit

```bash
pip install pre-commit
```

### 2. Create `.pre-commit-config.yaml`

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: check-added-large-files
      - id: check-json
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: detect-private-key

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.63.0
    hooks:
      - id: trufflehog
        args:
          - --no-update
          - --fail
```

### 3. Install hooks

```bash
pre-commit install
```

### 4. Test

```bash
pre-commit run --all-files
```

---

## CI/CD Integration

Add secret scanning to GitHub Actions:

```yaml
# .github/workflows/security.yml
name: Security Checks

on: [push, pull_request]

jobs:
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for better detection

      - name: TruffleHog Secret Scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
          extra_args: --only-verified

      - name: Detect Secrets
        uses: Yelp/detect-secrets-action@v0.1.0
```

---

## Best Practices

1. **Always commit .env.example**, never `.env`
2. **Use the hook**, don't bypass without review
3. **Update patterns** as new secret types are added
4. **Document bypasses** in commit messages if necessary
5. **Rotate keys** if accidentally committed (even if caught by hook)

---

## Resources

- [Git Hooks Documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)
- [Pre-commit Framework](https://pre-commit.com/)
- [TruffleHog (Secret Scanning)](https://github.com/trufflesecurity/trufflehog)
- [Detect Secrets (Yelp)](https://github.com/Yelp/detect-secrets)

---

**Last Updated**: 2025-11-08
**Maintainer**: Security Team

