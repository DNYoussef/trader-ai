#!/usr/bin/env python3
"""
Plaid API Client for Banking Integration
Handles Plaid Link token creation, account connections, and data fetching.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import plaid
from plaid.api import plaid_api
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
from plaid.model.accounts_get_request import AccountsGetRequest
from plaid.model.accounts_balance_get_request import AccountsBalanceGetRequest
from plaid.model.transactions_get_request import TransactionsGetRequest

logger = logging.getLogger(__name__)


@dataclass
class PlaidAccount:
    """Bank account data structure."""
    account_id: str
    name: str
    official_name: Optional[str]
    type: str
    subtype: str
    mask: Optional[str]
    current_balance: float
    available_balance: Optional[float]
    currency_code: str


@dataclass
class PlaidTransaction:
    """Transaction data structure."""
    transaction_id: str
    account_id: str
    amount: float
    date: str
    name: str
    merchant_name: Optional[str]
    category: List[str]
    pending: bool


class PlaidClient:
    """
    Plaid API client for bank account integration.

    Supports sandbox, development, and production environments.
    Handles Link token creation, public token exchange, and data fetching.
    """

    def __init__(self, client_id: str = None, secret: str = None, environment: str = "sandbox"):
        """
        Initialize Plaid client.

        Args:
            client_id: Plaid client ID (falls back to env var PLAID_CLIENT_ID)
            secret: Plaid secret (falls back to env var PLAID_SECRET)
            environment: Plaid environment (sandbox, development, production)
        """
        self.client_id = client_id or os.getenv("PLAID_CLIENT_ID")
        self.secret = secret or os.getenv("PLAID_SECRET")
        self.environment = environment

        if not self.client_id or not self.secret:
            raise ValueError(
                "Plaid credentials not provided. "
                "Set PLAID_CLIENT_ID and PLAID_SECRET environment variables "
                "or pass them to PlaidClient constructor."
            )

        # Map environment string to Plaid host
        env_map = {
            "sandbox": plaid.Environment.Sandbox,
            "development": plaid.Environment.Development,
            "production": plaid.Environment.Production
        }

        if environment not in env_map:
            raise ValueError(f"Invalid environment: {environment}. Must be sandbox, development, or production.")

        # Initialize Plaid configuration
        configuration = plaid.Configuration(
            host=env_map[environment],
            api_key={
                'clientId': self.client_id,
                'secret': self.secret,
            }
        )

        api_client = plaid.ApiClient(configuration)
        self.client = plaid_api.PlaidApi(api_client)

        logger.info(f"Plaid client initialized for {environment} environment")

    def create_link_token(self, user_id: str = "trader-ai-user") -> Dict[str, str]:
        """
        Create a Link token for Plaid Link initialization.

        Args:
            user_id: Unique user identifier

        Returns:
            Dict containing link_token and expiration

        Raises:
            plaid.ApiException: If Plaid API call fails
        """
        try:
            request = LinkTokenCreateRequest(
                user=LinkTokenCreateRequestUser(client_user_id=user_id),
                client_name="Trader AI",
                products=[Products("auth"), Products("transactions")],
                country_codes=[CountryCode("US")],
                language="en"
            )

            response = self.client.link_token_create(request)

            logger.info(f"Link token created for user {user_id}")

            return {
                "link_token": response['link_token'],
                "expiration": response['expiration'],
                "request_id": response['request_id']
            }

        except plaid.ApiException as e:
            logger.error(f"Failed to create link token: {e}")
            error_response = json.loads(e.body)
            raise Exception(f"Plaid API error: {error_response.get('error_message', str(e))}")

    def exchange_public_token(self, public_token: str) -> str:
        """
        Exchange public token for access token.

        This should be called after user completes Plaid Link flow.
        Store the returned access_token securely for future API calls.

        Args:
            public_token: Public token from Plaid Link

        Returns:
            Access token (store this securely!)

        Raises:
            plaid.ApiException: If exchange fails
        """
        try:
            request = ItemPublicTokenExchangeRequest(public_token=public_token)
            response = self.client.item_public_token_exchange(request)

            access_token = response['access_token']
            item_id = response['item_id']

            logger.info(f"Public token exchanged successfully. Item ID: {item_id}")

            return access_token

        except plaid.ApiException as e:
            logger.error(f"Failed to exchange public token: {e}")
            error_response = json.loads(e.body)
            raise Exception(f"Plaid API error: {error_response.get('error_message', str(e))}")

    def get_accounts(self, access_token: str) -> List[PlaidAccount]:
        """
        Fetch all linked bank accounts.

        Args:
            access_token: Access token from exchange_public_token()

        Returns:
            List of PlaidAccount objects

        Raises:
            plaid.ApiException: If API call fails
        """
        try:
            request = AccountsGetRequest(access_token=access_token)
            response = self.client.accounts_get(request)

            accounts = []
            for account in response['accounts']:
                accounts.append(PlaidAccount(
                    account_id=account['account_id'],
                    name=account['name'],
                    official_name=account.get('official_name'),
                    type=account['type'],
                    subtype=account['subtype'],
                    mask=account.get('mask'),
                    current_balance=account['balances']['current'],
                    available_balance=account['balances'].get('available'),
                    currency_code=account['balances']['iso_currency_code'] or 'USD'
                ))

            logger.info(f"Retrieved {len(accounts)} bank accounts")
            return accounts

        except plaid.ApiException as e:
            logger.error(f"Failed to get accounts: {e}")
            self._handle_api_error(e)

    def get_balances(self, access_token: str) -> List[Dict[str, Any]]:
        """
        Fetch real-time account balances.

        Args:
            access_token: Access token from exchange_public_token()

        Returns:
            List of account balance dicts

        Raises:
            plaid.ApiException: If API call fails
        """
        try:
            request = AccountsBalanceGetRequest(access_token=access_token)
            response = self.client.accounts_balance_get(request)

            balances = []
            for account in response['accounts']:
                balances.append({
                    'account_id': account['account_id'],
                    'name': account['name'],
                    'current': account['balances']['current'],
                    'available': account['balances'].get('available'),
                    'currency': account['balances']['iso_currency_code'] or 'USD',
                    'type': account['type'],
                    'subtype': account['subtype']
                })

            logger.info(f"Retrieved balances for {len(balances)} accounts")
            return balances

        except plaid.ApiException as e:
            logger.error(f"Failed to get balances: {e}")
            self._handle_api_error(e)

    def get_transactions(
        self,
        access_token: str,
        start_date: str = None,
        end_date: str = None,
        count: int = 100
    ) -> List[PlaidTransaction]:
        """
        Fetch transactions for specified date range.

        Args:
            access_token: Access token from exchange_public_token()
            start_date: Start date (YYYY-MM-DD) - defaults to 30 days ago
            end_date: End date (YYYY-MM-DD) - defaults to today
            count: Maximum number of transactions to fetch (default 100)

        Returns:
            List of PlaidTransaction objects

        Raises:
            plaid.ApiException: If API call fails
        """
        try:
            # Default date range: last 30 days
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            request = TransactionsGetRequest(
                access_token=access_token,
                start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
                end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
                options={"count": count}
            )

            response = self.client.transactions_get(request)

            transactions = []
            for txn in response['transactions']:
                transactions.append(PlaidTransaction(
                    transaction_id=txn['transaction_id'],
                    account_id=txn['account_id'],
                    amount=txn['amount'],
                    date=str(txn['date']),
                    name=txn['name'],
                    merchant_name=txn.get('merchant_name'),
                    category=txn.get('category', []),
                    pending=txn['pending']
                ))

            logger.info(f"Retrieved {len(transactions)} transactions from {start_date} to {end_date}")
            return transactions

        except plaid.ApiException as e:
            logger.error(f"Failed to get transactions: {e}")
            self._handle_api_error(e)

    def _handle_api_error(self, exception: plaid.ApiException):
        """
        Handle Plaid API exceptions with detailed error messages.

        Args:
            exception: Plaid API exception

        Raises:
            Exception: With detailed error message
        """
        try:
            error_response = json.loads(exception.body)
            error_code = error_response.get('error_code', 'UNKNOWN')
            error_message = error_response.get('error_message', str(exception))
            error_type = error_response.get('error_type', 'API_ERROR')

            # Handle specific error types
            if error_code == 'ITEM_LOGIN_REQUIRED':
                raise Exception(
                    "Bank login required. User needs to re-authenticate through Plaid Link."
                )
            elif error_code == 'RATE_LIMIT_EXCEEDED':
                raise Exception(
                    "Plaid API rate limit exceeded. Please retry in a few minutes."
                )
            elif error_type == 'INVALID_REQUEST':
                raise Exception(f"Invalid request: {error_message}")
            else:
                raise Exception(f"Plaid API error ({error_code}): {error_message}")

        except json.JSONDecodeError:
            raise Exception(f"Plaid API error: {str(exception)}")


# Utility functions for easy access
def create_plaid_client(config_path: str = None) -> PlaidClient:
    """
    Create PlaidClient from config file.

    Args:
        config_path: Path to config.json (defaults to config/config.json)

    Returns:
        Initialized PlaidClient
    """
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Handle env: prefix for credentials
    client_id = config.get('plaid_client_id', '')
    secret = config.get('plaid_secret', '')

    if client_id.startswith('env:'):
        client_id = os.getenv(client_id[4:])
    if secret.startswith('env:'):
        secret = os.getenv(secret[4:])

    return PlaidClient(
        client_id=client_id,
        secret=secret,
        environment=config.get('plaid_env', 'sandbox')
    )
