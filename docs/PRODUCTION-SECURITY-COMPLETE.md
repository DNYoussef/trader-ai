# Production Security Configuration - COMPLETE

## Summary

Production security features have been successfully configured for the trader-ai application:

## What Was Implemented

### 1. Rate Limiting (SlowAPI)
- **Location**: `src/security/security_middleware.py`
- **Configuration**: `config/config.json` → `rate_limit_per_minute: 10`
- **Features**:
  - IP-based rate limiting
  - Configurable limits per endpoint
  - Automatic 429 responses when limits exceeded
  - Rate limit headers in responses

**Applied to sensitive endpoints**:
- `/api/plaid/create_link_token` - 10 req/min
- `/api/plaid/exchange_public_token` - 10 req/min
- `/api/trade/execute` - 30 req/min
- WebSocket connections - 100/min

### 2. Security Headers
- **Location**: `src/security/security_middleware.py`
- **Enabled**: `security_headers_enabled: true` in config

**Headers added to all responses**:
- `Strict-Transport-Security` - Force HTTPS for 1 year
- `X-Content-Type-Options: nosniff` - Prevent MIME sniffing
- `X-Frame-Options: DENY` - Prevent clickjacking
- `X-XSS-Protection: 1; mode=block` - Enable XSS filter
- `Content-Security-Policy` - Restrict resource loading
- `Referrer-Policy` - Control referrer information
- `Permissions-Policy` - Disable unnecessary browser APIs

### 3. HTTPS Enforcement
- **Enabled**: When `environment: "production"` and `enforce_https: true`
- **Behavior**: All HTTP requests redirected to HTTPS (301 redirect)
- **Development**: Disabled to allow localhost testing

### 4. Database Encryption
- **Setup Script**: `scripts/setup_security.py`
- **Features**:
  - Fernet encryption key generation
  - JWT secret key generation
  - Database file permission setting (chmod 600)
  - .env file creation from template

### 5. Environment Configuration
- **Template**: `.env.example`
- **Variables**:
  - `PLAID_CLIENT_ID` - Plaid API credentials
  - `PLAID_SECRET` - Plaid API secret
  - `DATABASE_ENCRYPTION_KEY` - Fernet encryption key (auto-generated)
  - `JWT_SECRET_KEY` - JWT signing key (auto-generated)
  - `ENVIRONMENT` - development/production
  - `ENFORCE_HTTPS` - true/false
  - `RATE_LIMIT_PER_MINUTE` - Requests per minute

## Files Created/Modified

### New Files
1. `src/security/rate_limiter.py` - SlowAPI rate limiting wrapper
2. `src/security/security_middleware.py` - Comprehensive security middleware
3. `src/security/__init__.py` - Security module exports
4. `scripts/setup_security.py` - Security setup automation
5. `.env.example` - Environment variable template
6. `docs/SECURITY.md` - Complete security documentation

### Modified Files
1. `requirements.txt` - Added `slowapi>=0.1.9` and `cryptography>=41.0.0`
2. `config/config.json` - Added security configuration:
   ```json
   {
     "environment": "development",
     "enforce_https": false,
     "rate_limit_per_minute": 10,
     "security_headers_enabled": true
   }
   ```
3. `src/dashboard/run_server_simple.py` - Integrated security middleware

## Usage

### Development Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run security setup
python scripts/setup_security.py

# 3. Add Plaid credentials to .env
# Edit .env and replace placeholder values

# 4. Start server
cd src/dashboard
python run_server_simple.py
```

### Production Deployment
```bash
# 1. Update config for production
# Edit config/config.json:
{
  "environment": "production",
  "enforce_https": true,
  "rate_limit_per_minute": 10,
  "security_headers_enabled": true
}

# 2. Generate production keys
python scripts/setup_security.py

# 3. Secure database permissions
chmod 600 data/bank_accounts.db
chmod 700 data/

# 4. Set up SSL/TLS certificates
# Configure reverse proxy (nginx/Apache) with SSL

# 5. Start server
python src/dashboard/run_server_simple.py
```

### Testing Security Features

```bash
# Test rate limiting
for i in {1..15}; do curl http://localhost:8000/api/health; done
# Should see 429 after 10 requests

# Test security headers
curl -I http://localhost:8000/api/health
# Should see X-Frame-Options, CSP, etc.

# Test HTTPS redirect (production only)
curl -I http://yoursite.com/api/health
# Should see 301 → https://yoursite.com/api/health
```

## Security Configuration Reference

### Rate Limit Configuration
```python
# Custom rate limits in routes
@app.post("/api/custom")
@limiter.limit("20/minute")  # 20 requests per minute
async def custom_endpoint(request: Request):
    pass

# Use convenience decorators
from src.security import rate_limit_strict, rate_limit_moderate

@app.post("/api/strict")
@rate_limit_strict(limiter)  # 10/min
async def strict_endpoint(request: Request):
    pass
```

### Security Headers Customization
Modify `src/security/security_middleware.py`:

```python
# Add custom headers
response.headers["X-Custom-Header"] = "value"

# Modify CSP
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://trusted-cdn.com; "
    # ... other directives
)
```

### HTTPS Configuration
For reverse proxy (nginx):

```nginx
server {
    listen 443 ssl http2;
    server_name yoursite.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name yoursite.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Maintenance

### Logs to Monitor
- `logs/security_audit.log` - Security events
- `logs/rate_limit.log` - Rate limit violations
- `logs/server.log` - General server logs

### Regular Maintenance
1. **Rotate encryption keys** every 90 days
   ```bash
   python scripts/setup_security.py  # Generate new keys
   # Update .env with new keys
   # Restart application
   ```

2. **Review rate limits** monthly
   - Analyze rate limit violations
   - Adjust limits based on traffic patterns

3. **Update dependencies** weekly
   ```bash
   pip list --outdated
   pip-audit  # Check for vulnerabilities
   ```

4. **Test security headers** monthly
   - Use https://securityheaders.com
   - Verify CSP configuration
   - Check HSTS preload eligibility

## Production Checklist

- [x] Rate limiting configured
- [x] Security headers enabled
- [x] HTTPS enforcement ready (production)
- [x] Database encryption setup
- [x] Environment variables configured
- [x] Setup script created
- [x] Documentation complete
- [ ] SSL/TLS certificates installed (deployment)
- [ ] Firewall rules configured (deployment)
- [ ] Monitoring and alerting setup (deployment)
- [ ] Security audit completed (deployment)

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run setup script**: `python scripts/setup_security.py`
3. **Test locally**: Start server and verify rate limiting works
4. **Deploy to production**: Follow production deployment checklist
5. **Monitor**: Set up alerts for security events

## Support

For security issues or questions:
- Review: `docs/SECURITY.md`
- Contact: security@trader-ai.example.com
- GitHub: https://github.com/yourorg/trader-ai/security

## References

- SlowAPI: https://slowapi.readthedocs.io/
- OWASP Security Headers: https://owasp.org/www-project-secure-headers/
- FastAPI Security: https://fastapi.tiangolo.com/tutorial/security/
- Content Security Policy: https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP
