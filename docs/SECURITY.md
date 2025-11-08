# Security Configuration Guide

## Overview

Trader-AI implements multiple layers of security to protect sensitive data and prevent unauthorized access. This guide covers security setup, configuration, and best practices.

## Quick Start

### 1. Run Security Setup Script

```bash
python scripts/setup_security.py
```

This script will:
- Generate secure encryption and JWT keys
- Create `.env` file from template
- Set proper file permissions on databases
- Validate security configuration

### 2. Add Plaid Credentials

Edit `.env` and add your Plaid API credentials:

```bash
PLAID_CLIENT_ID=your_actual_client_id
PLAID_SECRET=your_actual_secret
```

### 3. Production Deployment

For production, update `config/config.json`:

```json
{
  "environment": "production",
  "enforce_https": true,
  "rate_limit_per_minute": 10,
  "security_headers_enabled": true
}
```

## Security Features

### 1. Rate Limiting

**Configuration**: `config.json` → `rate_limit_per_minute`

**Default Limits**:
- Plaid endpoints: 10 requests/minute per IP
- Trade execution: 30 requests/minute per IP
- General API: 60 requests/minute per IP
- WebSocket connections: 100/minute per IP

**Implementation**:
```python
from src.security import limiter, rate_limit_strict

@app.post("/api/sensitive")
@limiter.limit("10/minute")
async def sensitive_endpoint(request: Request):
    # Your code here
    pass
```

**Custom Limits**:
- `@rate_limit_strict()` - 10/min
- `@rate_limit_moderate()` - 30/min
- `@rate_limit_relaxed()` - 60/min
- `@rate_limit_websocket()` - 100/min

### 2. Security Headers

**Enabled by default** when `security_headers_enabled: true`

Headers automatically added to all HTTP responses:

| Header | Value | Purpose |
|--------|-------|---------|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | Force HTTPS for 1 year |
| `X-Content-Type-Options` | `nosniff` | Prevent MIME sniffing |
| `X-Frame-Options` | `DENY` | Prevent clickjacking |
| `X-XSS-Protection` | `1; mode=block` | Enable XSS filter |
| `Content-Security-Policy` | (see below) | Control resource loading |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Control referrer info |
| `Permissions-Policy` | `geolocation=(), microphone=(), camera=()` | Disable unnecessary APIs |

**Content Security Policy**:
```
default-src 'self';
script-src 'self' 'unsafe-inline' 'unsafe-eval';
style-src 'self' 'unsafe-inline';
img-src 'self' data: https:;
connect-src 'self' ws: wss:;
font-src 'self' data:;
```

### 3. HTTPS Enforcement

**Enabled in production** when `environment: "production"` and `enforce_https: true`

All HTTP requests are automatically redirected to HTTPS with 301 status code.

**Example**:
```
http://example.com/api/data → https://example.com/api/data (301 Redirect)
```

**Development Mode**: HTTPS enforcement is disabled to allow `localhost` testing without SSL certificates.

### 4. Database Encryption

**Encryption Keys**: Generated via `scripts/setup_security.py`

**File Permissions** (Unix/Linux):
```bash
chmod 600 data/bank_accounts.db  # Owner read/write only
chmod 700 data/                   # Owner access only
```

**Windows**: Use NTFS permissions to restrict access:
```powershell
icacls "data\bank_accounts.db" /inheritance:r /grant:r "$env:USERNAME:(R,W)"
```

### 5. Environment Variables

**Sensitive data** stored in `.env` file (never committed to git):

```bash
# Plaid API
PLAID_CLIENT_ID=...
PLAID_SECRET=...

# Encryption
DATABASE_ENCRYPTION_KEY=...  # Fernet key (44 chars base64)
JWT_SECRET_KEY=...            # URL-safe token (32+ chars)

# Environment
ENVIRONMENT=production
ENFORCE_HTTPS=true
```

**Loading in Python**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv('PLAID_CLIENT_ID')
secret = os.getenv('PLAID_SECRET')
```

## Security Validation

### Manual Validation

```bash
# Run security validation
python scripts/setup_security.py

# Check file permissions
ls -la data/
ls -la .env

# Verify rate limiting
curl -I http://localhost:8000/api/health
# Should include rate limit headers
```

### Automated Testing

```bash
# Run security tests
pytest tests/security/ -v

# Check for exposed secrets
git secrets --scan

# Audit dependencies
pip-audit

# Security linting
bandit -r src/
```

## Production Deployment Checklist

- [ ] Run `scripts/setup_security.py`
- [ ] Update `.env` with production credentials
- [ ] Set `environment: "production"` in `config.json`
- [ ] Enable `enforce_https: true`
- [ ] Configure rate limiting for your traffic
- [ ] Set up SSL/TLS certificates
- [ ] Restrict database file permissions (`chmod 600`)
- [ ] Enable audit logging
- [ ] Configure firewall rules
- [ ] Set up monitoring and alerting
- [ ] Review security headers in browser DevTools
- [ ] Test rate limiting with load testing
- [ ] Verify HTTPS redirect works
- [ ] Run `pytest tests/security/ -v`

## Monitoring and Logging

### Rate Limit Monitoring

Rate limit violations are logged:

```
[2025-11-07 12:34:56] WARNING Rate limit exceeded for IP 192.168.1.100 on /api/plaid/create_link_token
```

### Security Audit Log

All security events are logged to `logs/security_audit.log`:

```json
{
  "timestamp": "2025-11-07T12:34:56Z",
  "event": "rate_limit_exceeded",
  "ip": "192.168.1.100",
  "endpoint": "/api/plaid/create_link_token",
  "user_agent": "Mozilla/5.0..."
}
```

### Alerting

Configure alerts for:
- Repeated rate limit violations (potential attack)
- HTTPS enforcement failures
- Database permission changes
- Unusual API access patterns

## Common Issues

### 1. Rate Limit Too Restrictive

**Symptom**: Legitimate users getting rate limited

**Solution**: Increase `rate_limit_per_minute` in `config.json`:
```json
{
  "rate_limit_per_minute": 30  // Increase from 10
}
```

### 2. HTTPS Redirect Loop

**Symptom**: Infinite redirect when accessing site

**Solution**: Ensure reverse proxy (nginx/Apache) is forwarding HTTPS headers:
```nginx
proxy_set_header X-Forwarded-Proto $scheme;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
```

### 3. Database Permission Errors

**Symptom**: "Permission denied" when accessing database

**Solution**:
```bash
# Fix permissions
chmod 600 data/bank_accounts.db
chown $(whoami):$(whoami) data/bank_accounts.db
```

### 4. Missing Rate Limit Module

**Symptom**: `ImportError: No module named 'slowapi'`

**Solution**:
```bash
pip install -r requirements.txt
```

## Security Best Practices

1. **Never commit secrets**: Use `.env` and `.gitignore`
2. **Rotate keys regularly**: Update `DATABASE_ENCRYPTION_KEY` and `JWT_SECRET_KEY` every 90 days
3. **Use strong passwords**: Minimum 16 characters for production keys
4. **Monitor logs**: Set up alerting for security events
5. **Update dependencies**: Run `pip-audit` weekly
6. **Principle of least privilege**: Only grant necessary permissions
7. **Defense in depth**: Multiple security layers (rate limiting + headers + HTTPS)
8. **Regular security audits**: Run `bandit` and review findings

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [SlowAPI Documentation](https://slowapi.readthedocs.io/)
- [Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)
- [Cryptography Best Practices](https://cryptography.io/en/latest/)

## Support

For security issues, please email: security@trader-ai.example.com

**DO NOT** open public GitHub issues for security vulnerabilities.
