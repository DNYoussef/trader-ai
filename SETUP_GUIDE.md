# Complete Setup Guide - Gary×Taleb Trading System

## Table of Contents
1. [Prerequisites & Software Requirements](#prerequisites--software-requirements)
2. [Broker Account Setup](#broker-account-setup)
3. [API Keys & Configuration](#api-keys--configuration)
4. [Funding Your Account](#funding-your-account)
5. [Legal & Compliance](#legal--compliance)
6. [Installation Steps](#installation-steps)
7. [Money Movement](#money-movement)
8. [Security Best Practices](#security-best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Software Requirements

### Required Software Downloads

#### 1. Python (3.8 or higher)
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Mac**: `brew install python3` or download from python.org
- **Linux**: `sudo apt-get install python3 python3-pip`

#### 2. Node.js (16.0 or higher)
- **All Platforms**: Download from [nodejs.org](https://nodejs.org/)
- Includes npm (Node Package Manager)

#### 3. Git
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac**: `brew install git`
- **Linux**: `sudo apt-get install git`

#### 4. Visual Studio Code (Recommended)
- Download from [code.visualstudio.com](https://code.visualstudio.com/)

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **Internet**: Stable broadband connection (for trading)

---

## Broker Account Setup

### Alpaca Markets Account

#### 1. Create Account
1. Go to [alpaca.markets](https://alpaca.markets)
2. Click "Sign Up"
3. Choose account type:
   - **Paper Trading** (Recommended to start) - Free, unlimited
   - **Live Trading** - Requires identity verification

#### 2. Identity Verification (For Live Trading)
Required documents:
- Government-issued ID (Driver's License/Passport)
- Social Security Number (US residents)
- Proof of address (utility bill, bank statement)

#### 3. Account Approval
- Paper trading: Instant
- Live trading: 1-3 business days

### Alternative Brokers (Future Support)
- Interactive Brokers
- TD Ameritrade
- E*TRADE
- Robinhood (via unofficial API)

---

## API Keys & Configuration

### Getting Your Alpaca API Keys

#### Paper Trading Keys (Start Here)
1. Log into [app.alpaca.markets](https://app.alpaca.markets)
2. Select "Paper Trading" from the dropdown
3. Navigate to "API Keys" in the left sidebar
4. Click "Generate New Key"
5. Save both:
   - **API Key ID**: `PKxxxxxxxxxxxxxxxx`
   - **Secret Key**: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (shown only once!)

#### Live Trading Keys (After Testing)
1. Switch to "Live Trading" in the dropdown
2. Complete additional verification if required
3. Follow same steps as paper trading
4. **CRITICAL**: Keep these keys secure!

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Paper Trading Configuration (Start Here)
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Live Trading Configuration (After Testing)
# ALPACA_API_KEY=AKxxxxxxxxxxxxxxxx
# ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# ALPACA_BASE_URL=https://api.alpaca.markets

# Trading Configuration
TRADING_MODE=paper  # Change to 'live' when ready
INITIAL_CAPITAL=200.00
ENABLE_SIPHON=true
SIPHON_PERCENTAGE=0.5

# Risk Management
MAX_POSITION_SIZE=0.25
STOP_LOSS_PERCENTAGE=0.02
MAX_DAILY_LOSS=50.00

# Dashboard Configuration
DASHBOARD_PORT=3000
API_PORT=8000
ENABLE_WEBSOCKET=true
```

---

## Funding Your Account

### Paper Trading Account
- Automatically funded with $100,000 virtual money
- Reset anytime from dashboard
- No real money required

### Live Trading Account - Funding Methods

#### 1. ACH Transfer (Recommended - Free)
- **Setup**: Link bank account in Alpaca dashboard
- **Verification**: 2-3 business days (micro-deposits)
- **Transfer Time**: 3-5 business days
- **Minimum**: $1 (System requires $200 to start)
- **Maximum**: $50,000/day
- **Fees**: FREE

#### 2. Wire Transfer (Faster)
- **Setup**: Get wire instructions from Alpaca
- **Transfer Time**: Same day if before 4 PM ET
- **Minimum**: $500
- **Fees**: $10-30 (your bank) + $0 (Alpaca)

#### 3. ACAT Transfer (From Another Broker)
- **Transfer Time**: 5-7 business days
- **Minimum**: No minimum
- **Fees**: $0 (Alpaca covers)

### Step-by-Step ACH Setup
1. Log into Alpaca dashboard
2. Go to "Banking" → "Add Bank Account"
3. Enter bank routing and account numbers
4. Verify micro-deposits (2-3 days)
5. Initiate transfer → "Deposit Funds"
6. Enter amount ($200 minimum for system)
7. Confirm transfer

---

## Legal & Compliance

### Required Disclosures

#### Trading Authorization
- You must be 18+ years old
- US resident with valid SSN
- Not employed by FINRA member firm (or disclosed)

#### Tax Obligations
- **Form 1099-B**: Issued annually for trades
- **Wash Sale Rules**: Apply to losses
- **Pattern Day Trading**: $25,000 minimum for day trading
- Keep records of all transactions

#### Legal Structure Options

##### Individual Account (Simplest)
- Personal SSN
- Personal tax liability
- Easiest to set up

##### LLC Trading Account
1. Form LLC in your state (~$50-500)
2. Get EIN from IRS (free)
3. Open business bank account
4. Apply for Alpaca business account
5. Benefits: Liability protection, tax flexibility

##### Trust Account
- For estate planning
- Requires trust documents
- More complex setup

### Regulatory Compliance
- **Pattern Day Trading (PDT)**: Need $25,000 for 4+ day trades in 5 days
- **Reg T**: 50% margin requirement
- **SIPC Protection**: Up to $500,000 ($250,000 cash)

---

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/trader-ai.git
cd trader-ai
```

### 2. Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
# Navigate to dashboard
cd src/dashboard/frontend

# Install Node dependencies
npm install

# Build for production
npm run build
```

### 4. Database Setup (Optional)
```bash
# Install PostgreSQL (for trade history)
# Windows: Download installer from postgresql.org
# Mac: brew install postgresql
# Linux: sudo apt-get install postgresql

# Create database
createdb trader_ai
```

### 5. Launch System

#### Development Mode
```bash
# Terminal 1: Start backend
python main.py

# Terminal 2: Start dashboard backend
cd src/dashboard
python run_server_simple.py

# Terminal 3: Start dashboard frontend
cd src/dashboard/frontend
npm run dev
```

#### Production Mode
```bash
# Windows
start_ui.bat

# Linux/Mac
./start_ui.sh
```

---

## Money Movement

### Withdrawing Profits

#### Manual Withdrawal
1. Log into Alpaca dashboard
2. Go to "Banking" → "Withdraw Funds"
3. Select linked bank account
4. Enter amount
5. Confirm withdrawal
6. **Timeline**: 3-5 business days

#### Automated Siphon System
The system automatically manages profit withdrawal:

```python
# Every Friday at 6 PM ET:
if weekly_profit > 0:
    reinvest_amount = weekly_profit * 0.5  # 50% back into trading
    withdraw_amount = weekly_profit * 0.5  # 50% to your bank
```

To enable automated withdrawals:
1. Set `ENABLE_AUTO_WITHDRAW=true` in `.env`
2. Configure withdrawal schedule
3. Set minimum withdrawal amount ($100 recommended)

### Tax Considerations

#### Record Keeping
- Download monthly statements
- Export trade history CSV
- Track deposits/withdrawals
- Save for 7 years

#### Estimated Taxes
- Pay quarterly if expecting $1,000+ tax liability
- Due dates: April 15, June 15, Sept 15, Jan 15
- Set aside 25-35% of profits for taxes

---

## Security Best Practices

### API Key Security

#### Never Do This
```python
# BAD - Never hardcode keys
api_key = "PKxxxxxxxxxxxxx"  # NEVER DO THIS
```

#### Always Do This
```python
# GOOD - Use environment variables
import os
api_key = os.getenv('ALPACA_API_KEY')
```

### Additional Security Measures

1. **Two-Factor Authentication**
   - Enable on Alpaca account
   - Use authenticator app (not SMS)

2. **IP Whitelisting**
   - Restrict API access to your IP
   - Configure in Alpaca dashboard

3. **Encrypted Storage**
   - Use password manager for keys
   - Encrypt `.env` file when not in use

4. **Audit Logs**
   - System maintains logs in `.claude/.artifacts/`
   - Review weekly for anomalies

5. **Kill Switch**
   - Emergency stop: `Ctrl+C` twice
   - Stops all trading immediately

---

## Troubleshooting

### Common Issues & Solutions

#### Connection Issues
```bash
# Test API connection
python -c "
import alpaca_trade_api as tradeapi
api = tradeapi.REST(
    key_id='your_key',
    secret_key='your_secret',
    base_url='https://paper-api.alpaca.markets'
)
print(api.get_account())
"
```

#### Dashboard Not Loading
1. Check backend is running (port 8000)
2. Check frontend is running (port 3000/3001)
3. Clear browser cache
4. Check console for errors (F12)

#### Orders Not Executing
1. Check market hours (9:30 AM - 4:00 PM ET)
2. Verify account funding
3. Check for PDT restrictions
4. Review order size limits

#### WebSocket Disconnections
1. Check internet stability
2. Restart dashboard servers
3. Check Alpaca system status

### Getting Help

#### Support Channels
- **Alpaca Support**: support@alpaca.markets
- **Community Discord**: [discord.gg/alpaca](https://discord.gg/alpaca)
- **GitHub Issues**: Report bugs
- **Stack Overflow**: Tag with `alpaca-trade-api`

#### Emergency Contacts
- **Trading Issues**: Alpaca 24/7 support
- **Account Locked**: Call Alpaca directly
- **Suspicious Activity**: Change API keys immediately

---

## Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install Node.js 16+
- [ ] Install Git
- [ ] Create Alpaca account
- [ ] Get API keys (paper trading first)
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Configure `.env` file
- [ ] Test paper trading
- [ ] Fund account (if going live)
- [ ] Enable 2FA
- [ ] Set up tax withholding
- [ ] Configure kill switch
- [ ] Read risk disclosures
- [ ] Start with $200 minimum

---

## Risk Disclosure

**IMPORTANT**: Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The system's algorithms and strategies may not perform as expected in all market conditions.

**Never invest money you cannot afford to lose.**

**Start with paper trading until you fully understand the system.**

---

## Legal Notice

This software is provided "as is" without warranty of any kind. Users are responsible for:
- Compliance with all applicable laws and regulations
- Tax obligations
- Investment decisions and their consequences
- Securing their API keys and accounts

By using this system, you acknowledge that you understand and accept all risks involved in algorithmic trading.

---

*Last Updated: September 2024*
*Version: 1.0.0*