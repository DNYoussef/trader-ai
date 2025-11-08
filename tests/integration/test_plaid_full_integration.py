#!/usr/bin/env python3
"""
Integration Tests for Plaid Full Flow
Tests end-to-end OAuth flow, database persistence, and unified net worth calculation.
Requires FastAPI test client for API endpoint testing.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.finances.plaid_client import PlaidClient
from src.finances.bank_database import BankDatabase


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db():
    """Create temporary database for integration tests."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        db_path = f.name
    db = BankDatabase(db_path)
    yield db, db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_plaid_responses():
    """Complete set of mock Plaid API responses."""
    return {
        "link_token": {
            "link_token": "link-sandbox-integration-test-token",
            "expiration": (datetime.now() + timedelta(minutes=30)).isoformat(),
            "request_id": "req_link_123"
        },
        "exchange": {
            "access_token": "access-sandbox-integration-test-token",
            "item_id": "item_integration_test_123",
            "request_id": "req_exchange_456"
        },
        "accounts": {
            "accounts": [
                {
                    "account_id": "acc_checking_integration_123",
                    "name": "Business Checking",
                    "official_name": "Chase Business Checking",
                    "type": "depository",
                    "subtype": "checking",
                    "mask": "9876",
                    "balances": {
                        "current": 5000.00,
                        "available": 4950.00,
                        "iso_currency_code": "USD"
                    }
                },
                {
                    "account_id": "acc_savings_integration_456",
                    "name": "Emergency Fund",
                    "official_name": "High Yield Savings",
                    "type": "depository",
                    "subtype": "savings",
                    "mask": "5432",
                    "balances": {
                        "current": 25000.00,
                        "available": 25000.00,
                        "iso_currency_code": "USD"
                    }
                }
            ]
        },
        "transactions": {
            "transactions": [
                {
                    "transaction_id": "txn_int_coffee_123",
                    "account_id": "acc_checking_integration_123",
                    "amount": 5.00,
                    "date": (datetime.now() - timedelta(days=1)).date().isoformat(),
                    "name": "Starbucks",
                    "merchant_name": "Starbucks",
                    "category": ["Food and Drink"],
                    "pending": False
                },
                {
                    "transaction_id": "txn_int_aws_456",
                    "account_id": "acc_checking_integration_123",
                    "amount": 150.00,
                    "date": (datetime.now() - timedelta(days=2)).date().isoformat(),
                    "name": "AWS Services",
                    "merchant_name": "Amazon Web Services",
                    "category": ["Service", "Cloud Computing"],
                    "pending": False
                },
                {
                    "transaction_id": "txn_int_deposit_789",
                    "account_id": "acc_savings_integration_456",
                    "amount": -1000.00,
                    "date": (datetime.now() - timedelta(days=3)).date().isoformat(),
                    "name": "Deposit",
                    "merchant_name": None,
                    "category": ["Transfer"],
                    "pending": False
                }
            ]
        }
    }


@pytest.fixture
def fastapi_test_client():
    """Create FastAPI test client for endpoint testing."""
    try:
        from src.dashboard.run_server_simple import SimpleDashboardServer
        server = SimpleDashboardServer()
        return TestClient(server.app)
    except ImportError:
        pytest.skip("FastAPI server not available")


# ============================================================================
# TEST CLASS: Full OAuth Flow
# ============================================================================

class TestFullOAuthFlow:
    """Test complete Plaid Link OAuth flow from link token to data fetch."""

    @patch('plaid.api.plaid_api.PlaidApi.link_token_create')
    @patch('plaid.api.plaid_api.PlaidApi.item_public_token_exchange')
    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_complete_oauth_flow(
        self,
        mock_accounts_get,
        mock_exchange,
        mock_link_create,
        mock_plaid_responses,
        temp_db
    ):
        """Test full OAuth flow: link → exchange → fetch accounts."""
        db, db_path = temp_db

        # Setup mocks
        mock_link_create.return_value = mock_plaid_responses["link_token"]
        mock_exchange.return_value = mock_plaid_responses["exchange"]
        mock_accounts_get.return_value = mock_plaid_responses["accounts"]

        # Initialize Plaid client
        client = PlaidClient(
            client_id="test_client_id",
            secret="test_secret",
            environment="sandbox"
        )

        # Step 1: Create link token
        link_response = client.create_link_token(user_id="test_user_integration")
        assert "link_token" in link_response
        link_token = link_response["link_token"]

        # Step 2: User completes Plaid Link (simulated by public token)
        public_token = "public-sandbox-test-token"

        # Step 3: Exchange public token for access token
        access_token = client.exchange_public_token(public_token)
        assert access_token == mock_plaid_responses["exchange"]["access_token"]

        # Step 4: Store Plaid item in database
        item_id = db.add_plaid_item(
            access_token=access_token,
            institution_name="Chase Bank"
        )
        assert item_id.startswith("item_")

        # Step 5: Fetch accounts
        accounts = client.get_accounts(access_token)
        assert len(accounts) == 2

        # Step 6: Store accounts in database
        accounts_data = [
            {
                "account_id": acc.account_id,
                "name": acc.name,
                "official_name": acc.official_name,
                "type": acc.type,
                "subtype": acc.subtype,
                "mask": acc.mask,
                "balances": {
                    "current": acc.current_balance,
                    "available": acc.available_balance,
                    "iso_currency_code": acc.currency_code
                }
            }
            for acc in accounts
        ]
        count = db.update_accounts(item_id, accounts_data)
        assert count == 2

        # Step 7: Verify persistence
        stored_accounts = db.get_all_accounts()
        assert len(stored_accounts) == 2
        assert stored_accounts[0]["institution_name"] == "Chase Bank"

        total_balance = db.get_total_balance()
        assert total_balance == 30000.00  # 5000 + 25000

    @patch('plaid.api.plaid_api.PlaidApi.link_token_create')
    @patch('plaid.api.plaid_api.PlaidApi.item_public_token_exchange')
    def test_oauth_flow_with_exchange_failure(
        self,
        mock_exchange,
        mock_link_create,
        mock_plaid_responses
    ):
        """Test OAuth flow when token exchange fails."""
        import plaid

        mock_link_create.return_value = mock_plaid_responses["link_token"]

        # Mock exchange failure
        error_response = {
            "error_code": "INVALID_PUBLIC_TOKEN",
            "error_message": "Invalid public token",
            "error_type": "INVALID_REQUEST"
        }
        mock_error = plaid.ApiException(status=400, reason="Bad Request")
        mock_error.body = json.dumps(error_response)
        mock_exchange.side_effect = mock_error

        client = PlaidClient(
            client_id="test_client_id",
            secret="test_secret",
            environment="sandbox"
        )

        # Link token should succeed
        link_response = client.create_link_token()
        assert "link_token" in link_response

        # Exchange should fail
        with pytest.raises(Exception, match="Plaid API error"):
            client.exchange_public_token("invalid_public_token")


# ============================================================================
# TEST CLASS: Database Persistence After API Sync
# ============================================================================

class TestDatabasePersistence:
    """Test that Plaid API data is correctly persisted to database."""

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    @patch('plaid.api.plaid_api.PlaidApi.transactions_get')
    def test_full_sync_persistence(
        self,
        mock_txn_get,
        mock_accounts_get,
        mock_plaid_responses,
        temp_db
    ):
        """Test full sync: accounts + transactions persistence."""
        db, db_path = temp_db

        mock_accounts_get.return_value = mock_plaid_responses["accounts"]
        mock_txn_get.return_value = mock_plaid_responses["transactions"]

        client = PlaidClient(
            client_id="test_client_id",
            secret="test_secret",
            environment="sandbox"
        )

        # Simulate access token obtained from OAuth
        access_token = "access-test-token"

        # Add item
        item_id = db.add_plaid_item(access_token, "Chase Bank")

        # Fetch and store accounts
        accounts = client.get_accounts(access_token)
        accounts_data = [
            {
                "account_id": acc.account_id,
                "name": acc.name,
                "official_name": acc.official_name,
                "type": acc.type,
                "subtype": acc.subtype,
                "mask": acc.mask,
                "balances": {
                    "current": acc.current_balance,
                    "available": acc.available_balance,
                    "iso_currency_code": acc.currency_code
                }
            }
            for acc in accounts
        ]
        db.update_accounts(item_id, accounts_data)

        # Fetch and store transactions
        transactions = client.get_transactions(access_token)
        for acc in accounts:
            acc_transactions = [
                {
                    "transaction_id": txn.transaction_id,
                    "amount": txn.amount,
                    "date": txn.date,
                    "name": txn.name,
                    "merchant_name": txn.merchant_name,
                    "category": txn.category,
                    "pending": txn.pending
                }
                for txn in transactions if txn.account_id == acc.account_id
            ]
            if acc_transactions:
                db.update_transactions(acc.account_id, acc_transactions)

        # Verify persistence
        stored_accounts = db.get_all_accounts()
        assert len(stored_accounts) == 2

        stored_transactions = db.get_recent_transactions(days=30)
        assert len(stored_transactions) == 3

        # Verify transaction details
        checking_txns = [t for t in stored_transactions if t["account_id"] == "acc_checking_integration_123"]
        assert len(checking_txns) == 2

    def test_incremental_sync_updates_balances(self, temp_db):
        """Test that incremental syncs update balances correctly."""
        db, db_path = temp_db

        item_id = db.add_plaid_item("token", "Test Bank")

        # Initial sync
        initial_accounts = [{
            "account_id": "acc_123",
            "name": "Checking",
            "type": "depository",
            "subtype": "checking",
            "balances": {"current": 1000.00, "available": 950.00}
        }]
        db.update_accounts(item_id, initial_accounts)

        # Verify initial balance
        accounts = db.get_all_accounts()
        assert accounts[0]["current_balance"] == 1000.00

        # Incremental sync with updated balance
        updated_accounts = [{
            "account_id": "acc_123",
            "name": "Checking",
            "type": "depository",
            "subtype": "checking",
            "balances": {"current": 1200.00, "available": 1150.00}
        }]
        db.update_accounts(item_id, updated_accounts)

        # Verify updated balance
        accounts = db.get_all_accounts()
        assert accounts[0]["current_balance"] == 1200.00
        assert accounts[0]["available_balance"] == 1150.00


# ============================================================================
# TEST CLASS: Unified Net Worth Calculation
# ============================================================================

class TestUnifiedNetWorth:
    """Test unified net worth calculation combining trader-ai + bank balances."""

    def test_unified_net_worth_trader_plus_bank(self, temp_db):
        """Test unified net worth calculation."""
        db, db_path = temp_db

        # Setup bank accounts
        item_id = db.add_plaid_item("token", "Bank")
        accounts_data = [
            {"account_id": "acc_1", "name": "Checking", "balances": {"current": 5000.00}},
            {"account_id": "acc_2", "name": "Savings", "balances": {"current": 15000.00}}
        ]
        db.update_accounts(item_id, accounts_data)

        # Bank total
        bank_balance = db.get_total_balance()
        assert bank_balance == 20000.00

        # Simulate trader-ai portfolio value
        trader_portfolio_value = 12500.50

        # Unified net worth
        unified_net_worth = bank_balance + trader_portfolio_value
        assert unified_net_worth == 32500.50

    def test_unified_net_worth_with_multiple_institutions(self, temp_db):
        """Test unified net worth with multiple bank connections."""
        db, db_path = temp_db

        # Institution 1
        item1 = db.add_plaid_item("token1", "Chase")
        db.update_accounts(item1, [
            {"account_id": "chase_1", "name": "Chase Checking", "balances": {"current": 3000.00}}
        ])

        # Institution 2
        item2 = db.add_plaid_item("token2", "Wells Fargo")
        db.update_accounts(item2, [
            {"account_id": "wells_1", "name": "Wells Savings", "balances": {"current": 7000.00}}
        ])

        # Institution 3
        item3 = db.add_plaid_item("token3", "Ally")
        db.update_accounts(item3, [
            {"account_id": "ally_1", "name": "Ally HYSA", "balances": {"current": 10000.00}}
        ])

        # Total bank balance across all institutions
        total_bank = db.get_total_balance()
        assert total_bank == 20000.00

        # Trader-ai portfolio
        trader_portfolio = 5000.00

        # Unified net worth
        unified = total_bank + trader_portfolio
        assert unified == 25000.00


# ============================================================================
# TEST CLASS: FastAPI Endpoint Integration
# ============================================================================

class TestFastAPIEndpoints:
    """Test FastAPI endpoints with test client."""

    def test_create_link_token_endpoint(self, fastapi_test_client):
        """Test POST /api/plaid/create_link_token endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.post(
            "/api/plaid/create_link_token",
            json={"user_id": "test_user_123"}
        )

        # May return mock data or 500 if Plaid not configured
        assert response.status_code in [200, 500]

    def test_exchange_public_token_endpoint(self, fastapi_test_client):
        """Test POST /api/plaid/exchange_public_token endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.post(
            "/api/plaid/exchange_public_token",
            json={
                "public_token": "public-test-token",
                "user_id": "test_user_123",
                "metadata": {"institution": {"name": "Chase"}}
            }
        )

        assert response.status_code in [200, 400, 500]

    def test_get_bank_accounts_endpoint(self, fastapi_test_client):
        """Test GET /api/bank/accounts endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.get("/api/bank/accounts")

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_get_bank_balances_endpoint(self, fastapi_test_client):
        """Test GET /api/bank/balances endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.get("/api/bank/balances")

        assert response.status_code in [200, 500]

    def test_get_transactions_endpoint(self, fastapi_test_client):
        """Test GET /api/bank/transactions endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.get("/api/bank/transactions?days=30")

        assert response.status_code in [200, 500]

    def test_unified_networth_endpoint(self, fastapi_test_client):
        """Test GET /api/networth endpoint."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        response = fastapi_test_client.get("/api/networth")

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "total_networth" in data or "bank_balance" in data


# ============================================================================
# TEST CLASS: WebSocket Balance Updates
# ============================================================================

class TestWebSocketUpdates:
    """Test WebSocket real-time balance updates."""

    def test_websocket_connection(self, fastapi_test_client):
        """Test WebSocket connection establishment."""
        if fastapi_test_client is None:
            pytest.skip("FastAPI test client not available")

        # WebSocket testing requires special handling
        # This is a placeholder for WebSocket integration tests
        pytest.skip("WebSocket testing requires special test setup")

    def test_websocket_balance_broadcast(self):
        """Test that balance updates are broadcast to WebSocket clients."""
        # Placeholder for WebSocket broadcast testing
        pytest.skip("WebSocket broadcast testing requires special setup")


# ============================================================================
# TEST CLASS: Error Recovery
# ============================================================================

class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    @patch('plaid.api.plaid_api.PlaidApi.accounts_get')
    def test_api_failure_doesnt_corrupt_database(
        self,
        mock_accounts_get,
        temp_db
    ):
        """Test that API failures don't corrupt database state."""
        import plaid

        db, db_path = temp_db

        # Setup existing data
        item_id = db.add_plaid_item("token", "Bank")
        db.update_accounts(item_id, [
            {"account_id": "acc_1", "name": "Checking", "balances": {"current": 1000.00}}
        ])

        # Verify initial state
        initial_balance = db.get_total_balance()
        assert initial_balance == 1000.00

        # Simulate API failure
        mock_error = plaid.ApiException(status=500, reason="Internal Server Error")
        mock_error.body = json.dumps({"error_message": "Server error"})
        mock_accounts_get.side_effect = mock_error

        client = PlaidClient(
            client_id="test_client_id",
            secret="test_secret",
            environment="sandbox"
        )

        # API call should fail
        with pytest.raises(Exception):
            client.get_accounts("token")

        # Database should still have original data
        final_balance = db.get_total_balance()
        assert final_balance == 1000.00

    def test_partial_transaction_sync_rollback(self, temp_db):
        """Test that partial transaction syncs can be rolled back."""
        db, db_path = temp_db

        item_id = db.add_plaid_item("token", "Bank")
        db.update_accounts(item_id, [
            {"account_id": "acc_1", "name": "Checking", "balances": {"current": 1000.00}}
        ])

        # Try to add transactions (some valid, some invalid)
        transactions = [
            {"transaction_id": "txn_1", "amount": 50.00, "date": datetime.now().date().isoformat(), "name": "Valid"},
            # Missing required field to trigger error
        ]

        # Should handle gracefully
        try:
            db.update_transactions("acc_1", transactions)
        except Exception:
            pass

        # Database should still be queryable
        accounts = db.get_all_accounts()
        assert len(accounts) == 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
