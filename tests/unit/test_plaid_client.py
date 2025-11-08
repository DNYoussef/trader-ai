#!/usr/bin/env python3
"""
Unit Tests for PlaidClient
Tests Plaid API integration with mocked responses for offline development.
Target: 90%+ code coverage
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import asdict

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.finances.plaid_client import (
    PlaidClient,
    PlaidAccount,
    PlaidTransaction,
    create_plaid_client
)


# ============================================================================
# FIXTURES - Mock Plaid API Responses
# ============================================================================

@pytest.fixture
def mock_plaid_config():
    """Mock Plaid configuration."""
    return {
        "plaid_client_id": "test_client_id_123",
        "plaid_secret": "test_secret_456",
        "plaid_env": "sandbox"
    }


@pytest.fixture
def mock_link_token_response():
    """Mock Plaid Link token creation response."""
    return {
        "link_token": "link-sandbox-test-token-123",
        "expiration": (datetime.now() + timedelta(minutes=30)).isoformat(),
        "request_id": "req_test_123"
    }


@pytest.fixture
def mock_exchange_token_response():
    """Mock public token exchange response."""
    return {
        "access_token": "access-sandbox-test-token-456",
        "item_id": "item_test_789",
        "request_id": "req_test_456"
    }


@pytest.fixture
def mock_accounts_response():
    """Mock accounts get response with 2 accounts."""
    return {
        "accounts": [
            {
                "account_id": "acc_checking_123",
                "name": "Premium Checking",
                "official_name": "Chase Premium Checking Account",
                "type": "depository",
                "subtype": "checking",
                "mask": "1234",
                "balances": {
                    "current": 2500.50,
                    "available": 2450.00,
                    "iso_currency_code": "USD"
                }
            },
            {
                "account_id": "acc_savings_456",
                "name": "High Yield Savings",
                "official_name": None,
                "type": "depository",
                "subtype": "savings",
                "mask": "5678",
                "balances": {
                    "current": 10000.00,
                    "available": 10000.00,
                    "iso_currency_code": "USD"
                }
            }
        ],
        "item": {"item_id": "item_test_789"},
        "request_id": "req_test_789"
    }


@pytest.fixture
def mock_transactions_response():
    """Mock transactions get response with 3 transactions."""
    return {
        "transactions": [
            {
                "transaction_id": "txn_coffee_123",
                "account_id": "acc_checking_123",
                "amount": 4.50,
                "date": "2025-01-10",
                "name": "Starbucks",
                "merchant_name": "Starbucks Corporation",
                "category": ["Food and Drink", "Restaurants", "Coffee Shop"],
                "pending": False
            },
            {
                "transaction_id": "txn_grocery_456",
                "account_id": "acc_checking_123",
                "amount": 125.00,
                "date": "2025-01-09",
                "name": "Whole Foods",
                "merchant_name": "Whole Foods Market",
                "category": ["Shops", "Food and Beverage Store", "Supermarkets and Groceries"],
                "pending": False
            },
            {
                "transaction_id": "txn_transfer_789",
                "account_id": "acc_checking_123",
                "amount": -500.00,
                "date": "2025-01-08",
                "name": "Transfer to Savings",
                "merchant_name": None,
                "category": ["Transfer"],
                "pending": True
            }
        ],
        "total_transactions": 3,
        "request_id": "req_test_101"
    }


@pytest.fixture
def mock_error_response():
    """Mock Plaid API error response."""
    return {
        "error_code": "ITEM_LOGIN_REQUIRED",
        "error_message": "The user must authenticate again through Plaid Link",
        "error_type": "ITEM_ERROR",
        "request_id": "req_error_123"
    }


# ============================================================================
# TEST CLASS: PlaidClient Initialization
# ============================================================================

class TestPlaidClientInitialization:
    """Test PlaidClient initialization and configuration."""

    def test_init_with_credentials(self, mock_plaid_config):
        """Test PlaidClient initialization with explicit credentials."""
        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        assert client.client_id == mock_plaid_config["plaid_client_id"]
        assert client.secret == mock_plaid_config["plaid_secret"]
        assert client.environment == "sandbox"
        assert client.client is not None

    def test_init_with_env_vars(self, monkeypatch, mock_plaid_config):
        """Test PlaidClient initialization with environment variables."""
        monkeypatch.setenv("PLAID_CLIENT_ID", mock_plaid_config["plaid_client_id"])
        monkeypatch.setenv("PLAID_SECRET", mock_plaid_config["plaid_secret"])

        client = PlaidClient(environment="sandbox")

        assert client.client_id == mock_plaid_config["plaid_client_id"]
        assert client.secret == mock_plaid_config["plaid_secret"]

    def test_init_without_credentials_raises_error(self):
        """Test that missing credentials raises ValueError."""
        with pytest.raises(ValueError, match="Plaid credentials not provided"):
            PlaidClient(client_id=None, secret=None)

    def test_init_invalid_environment_raises_error(self, mock_plaid_config):
        """Test that invalid environment raises ValueError."""
        with pytest.raises(ValueError, match="Invalid environment"):
            PlaidClient(
                client_id=mock_plaid_config["plaid_client_id"],
                secret=mock_plaid_config["plaid_secret"],
                environment="invalid_env"
            )

    @pytest.mark.parametrize("environment", ["sandbox", "development", "production"])
    def test_init_with_all_environments(self, mock_plaid_config, environment):
        """Test initialization with all valid environments."""
        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment=environment
        )
        assert client.environment == environment


# ============================================================================
# TEST CLASS: Link Token Operations
# ============================================================================

class TestLinkTokenOperations:
    """Test Link token creation for Plaid Link initialization."""

    @patch('plaid.api.plaid_api.PlaidApi.link_token_create')
    def test_create_link_token_success(self, mock_link_create, mock_plaid_config, mock_link_token_response):
        """Test successful link token creation."""
        mock_link_create.return_value = mock_link_token_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        result = client.create_link_token(user_id="test_user_123")

        assert result["link_token"] == mock_link_token_response["link_token"]
        assert "expiration" in result
        assert "request_id" in result
        mock_link_create.assert_called_once()

    @patch('plaid.api.plaid_api.PlaidApi.link_token_create')
    def test_create_link_token_default_user_id(self, mock_link_create, mock_plaid_config, mock_link_token_response):
        """Test link token creation with default user_id."""
        mock_link_create.return_value = mock_link_token_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        result = client.create_link_token()
        assert result["link_token"] is not None

    @patch('plaid.api.plaid_api.PlaidApi.link_token_create')
    def test_create_link_token_api_error(self, mock_link_create, mock_plaid_config, mock_error_response):
        """Test link token creation with API error."""
        import plaid
        mock_error = plaid.ApiException(status=400, reason="Bad Request")
        mock_error.body = json.dumps(mock_error_response)
        mock_link_create.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="Plaid API error"):
            client.create_link_token()


# ============================================================================
# TEST CLASS: Public Token Exchange
# ============================================================================

class TestPublicTokenExchange:
    """Test public token exchange for access token."""

    @patch('plaid.api.plaid_api.PlaidApi.item_public_token_exchange')
    def test_exchange_public_token_success(self, mock_exchange, mock_plaid_config, mock_exchange_token_response):
        """Test successful public token exchange."""
        mock_exchange.return_value = mock_exchange_token_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        access_token = client.exchange_public_token("public-sandbox-test-token")

        assert access_token == mock_exchange_token_response["access_token"]
        mock_exchange.assert_called_once()

    @patch('plaid.api.plaid_api.PlaidApi.item_public_token_exchange')
    def test_exchange_invalid_token_raises_error(self, mock_exchange, mock_plaid_config):
        """Test exchange with invalid public token."""
        import plaid
        error_response = {
            "error_code": "INVALID_PUBLIC_TOKEN",
            "error_message": "The provided public token is invalid",
            "error_type": "INVALID_REQUEST"
        }
        mock_error = plaid.ApiException(status=400, reason="Bad Request")
        mock_error.body = json.dumps(error_response)
        mock_exchange.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="Plaid API error"):
            client.exchange_public_token("invalid_token")


# ============================================================================
# TEST CLASS: Account Operations
# ============================================================================

class TestAccountOperations:
    """Test account fetching and balance operations."""

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_get_accounts_success(self, mock_accounts_get, mock_plaid_config, mock_accounts_response):
        """Test successful account retrieval."""
        mock_accounts_get.return_value = mock_accounts_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        accounts = client.get_accounts("access-token-test")

        assert len(accounts) == 2
        assert isinstance(accounts[0], PlaidAccount)
        assert accounts[0].account_id == "acc_checking_123"
        assert accounts[0].current_balance == 2500.50
        assert accounts[0].name == "Premium Checking"
        assert accounts[1].account_id == "acc_savings_456"

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_get_accounts_empty_response(self, mock_accounts_get, mock_plaid_config):
        """Test account retrieval with no accounts."""
        mock_accounts_get.return_value = {"accounts": [], "item": {"item_id": "test"}}

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        accounts = client.get_accounts("access-token-test")
        assert len(accounts) == 0

    @patch('plaid.api.plaid_api.PlaidApi.accounts_balance_get')
    def test_get_balances_success(self, mock_balance_get, mock_plaid_config, mock_accounts_response):
        """Test successful balance retrieval."""
        mock_balance_get.return_value = mock_accounts_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        balances = client.get_balances("access-token-test")

        assert len(balances) == 2
        assert balances[0]["account_id"] == "acc_checking_123"
        assert balances[0]["current"] == 2500.50
        assert balances[0]["available"] == 2450.00
        assert balances[0]["currency"] == "USD"

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_get_accounts_null_currency_defaults_to_usd(self, mock_accounts_get, mock_plaid_config):
        """Test that null currency code defaults to USD."""
        response = {
            "accounts": [{
                "account_id": "acc_123",
                "name": "Test Account",
                "type": "depository",
                "subtype": "checking",
                "balances": {
                    "current": 1000.00,
                    "iso_currency_code": None  # Null currency
                }
            }]
        }
        mock_accounts_get.return_value = response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        accounts = client.get_accounts("access-token-test")
        assert accounts[0].currency_code == "USD"


# ============================================================================
# TEST CLASS: Transaction Operations
# ============================================================================

class TestTransactionOperations:
    """Test transaction fetching and date range handling."""

    @patch('plaid.api.plaid_api.PlaidApi.transactions_get')
    def test_get_transactions_success(self, mock_txn_get, mock_plaid_config, mock_transactions_response):
        """Test successful transaction retrieval."""
        mock_txn_get.return_value = mock_transactions_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        transactions = client.get_transactions(
            "access-token-test",
            start_date="2025-01-01",
            end_date="2025-01-15"
        )

        assert len(transactions) == 3
        assert isinstance(transactions[0], PlaidTransaction)
        assert transactions[0].transaction_id == "txn_coffee_123"
        assert transactions[0].amount == 4.50
        assert transactions[0].merchant_name == "Starbucks Corporation"

    @patch('plaid.api.plaid_api.PlaidApi.transactions_get')
    def test_get_transactions_default_date_range(self, mock_txn_get, mock_plaid_config, mock_transactions_response):
        """Test transactions with default 30-day date range."""
        mock_txn_get.return_value = mock_transactions_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        transactions = client.get_transactions("access-token-test")

        # Should default to last 30 days
        assert len(transactions) == 3
        mock_txn_get.assert_called_once()

    @patch('plaid.api.plaid_api.PlaidApi.transactions_get')
    def test_get_transactions_custom_count(self, mock_txn_get, mock_plaid_config, mock_transactions_response):
        """Test transactions with custom count parameter."""
        mock_txn_get.return_value = mock_transactions_response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        transactions = client.get_transactions(
            "access-token-test",
            count=50
        )

        assert len(transactions) == 3

    @patch('plaid.api.plaid_api.PlaidApi.transactions_get')
    def test_get_transactions_empty_category(self, mock_txn_get, mock_plaid_config):
        """Test transaction with empty category list."""
        response = {
            "transactions": [{
                "transaction_id": "txn_123",
                "account_id": "acc_123",
                "amount": 10.00,
                "date": "2025-01-10",
                "name": "Test",
                "category": [],  # Empty category
                "pending": False
            }]
        }
        mock_txn_get.return_value = response

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        transactions = client.get_transactions("access-token-test")
        assert transactions[0].category == []


# ============================================================================
# TEST CLASS: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_handle_item_login_required_error(self, mock_accounts_get, mock_plaid_config):
        """Test ITEM_LOGIN_REQUIRED error handling."""
        import plaid
        error_response = {
            "error_code": "ITEM_LOGIN_REQUIRED",
            "error_message": "Login required",
            "error_type": "ITEM_ERROR"
        }
        mock_error = plaid.ApiException(status=400, reason="Bad Request")
        mock_error.body = json.dumps(error_response)
        mock_accounts_get.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="Bank login required"):
            client.get_accounts("access-token-test")

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_handle_rate_limit_error(self, mock_accounts_get, mock_plaid_config):
        """Test RATE_LIMIT_EXCEEDED error handling."""
        import plaid
        error_response = {
            "error_code": "RATE_LIMIT_EXCEEDED",
            "error_message": "Rate limit exceeded",
            "error_type": "RATE_LIMIT_ERROR"
        }
        mock_error = plaid.ApiException(status=429, reason="Too Many Requests")
        mock_error.body = json.dumps(error_response)
        mock_accounts_get.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="rate limit exceeded"):
            client.get_accounts("access-token-test")

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_handle_invalid_request_error(self, mock_accounts_get, mock_plaid_config):
        """Test INVALID_REQUEST error handling."""
        import plaid
        error_response = {
            "error_code": "INVALID_ACCESS_TOKEN",
            "error_message": "Invalid access token",
            "error_type": "INVALID_REQUEST"
        }
        mock_error = plaid.ApiException(status=400, reason="Bad Request")
        mock_error.body = json.dumps(error_response)
        mock_accounts_get.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="Invalid request"):
            client.get_accounts("access-token-test")

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_handle_generic_api_error(self, mock_accounts_get, mock_plaid_config):
        """Test generic API error handling."""
        import plaid
        mock_error = plaid.ApiException(status=500, reason="Internal Server Error")
        mock_error.body = "Not JSON"
        mock_accounts_get.side_effect = mock_error

        client = PlaidClient(
            client_id=mock_plaid_config["plaid_client_id"],
            secret=mock_plaid_config["plaid_secret"],
            environment="sandbox"
        )

        with pytest.raises(Exception, match="Plaid API error"):
            client.get_accounts("access-token-test")


# ============================================================================
# TEST CLASS: Dataclass Structures
# ============================================================================

class TestDataclassStructures:
    """Test PlaidAccount and PlaidTransaction dataclass structures."""

    def test_plaid_account_creation(self):
        """Test PlaidAccount dataclass instantiation."""
        account = PlaidAccount(
            account_id="acc_123",
            name="Test Checking",
            official_name="Official Test Checking",
            type="depository",
            subtype="checking",
            mask="1234",
            current_balance=1500.00,
            available_balance=1400.00,
            currency_code="USD"
        )

        assert account.account_id == "acc_123"
        assert account.current_balance == 1500.00
        assert account.currency_code == "USD"

    def test_plaid_account_to_dict(self):
        """Test PlaidAccount conversion to dictionary."""
        account = PlaidAccount(
            account_id="acc_123",
            name="Test",
            official_name=None,
            type="depository",
            subtype="checking",
            mask="1234",
            current_balance=1000.00,
            available_balance=950.00,
            currency_code="USD"
        )

        account_dict = asdict(account)
        assert account_dict["account_id"] == "acc_123"
        assert account_dict["official_name"] is None

    def test_plaid_transaction_creation(self):
        """Test PlaidTransaction dataclass instantiation."""
        txn = PlaidTransaction(
            transaction_id="txn_123",
            account_id="acc_123",
            amount=50.00,
            date="2025-01-10",
            name="Test Purchase",
            merchant_name="Test Merchant",
            category=["Shopping"],
            pending=False
        )

        assert txn.transaction_id == "txn_123"
        assert txn.amount == 50.00
        assert txn.pending is False

    def test_plaid_transaction_to_dict(self):
        """Test PlaidTransaction conversion to dictionary."""
        txn = PlaidTransaction(
            transaction_id="txn_123",
            account_id="acc_123",
            amount=50.00,
            date="2025-01-10",
            name="Test",
            merchant_name=None,
            category=[],
            pending=True
        )

        txn_dict = asdict(txn)
        assert txn_dict["pending"] is True
        assert txn_dict["merchant_name"] is None


# ============================================================================
# TEST CLASS: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Test module-level utility functions."""

    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_create_plaid_client_from_config(self, mock_json_load, mock_open, mock_plaid_config):
        """Test create_plaid_client() utility function."""
        mock_json_load.return_value = mock_plaid_config

        client = create_plaid_client()

        assert isinstance(client, PlaidClient)
        assert client.client_id == mock_plaid_config["plaid_client_id"]

    @patch('builtins.open', create=True)
    @patch('json.load')
    def test_create_plaid_client_with_env_prefix(self, mock_json_load, mock_open, monkeypatch):
        """Test config with env: prefix."""
        monkeypatch.setenv("PLAID_CLIENT_ID", "test_id")
        monkeypatch.setenv("PLAID_SECRET", "test_secret")

        mock_json_load.return_value = {
            "plaid_client_id": "env:PLAID_CLIENT_ID",
            "plaid_secret": "env:PLAID_SECRET",
            "plaid_env": "sandbox"
        }

        client = create_plaid_client()
        assert client.client_id == "test_id"
        assert client.secret == "test_secret"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=src.finances.plaid_client", "--cov-report=term-missing"])
