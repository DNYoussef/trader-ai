/**
 * Unit Tests for PlaidLinkButton Component
 * Tests Plaid Link flow, token exchange, and error handling
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import axios from 'axios';
import { PlaidLinkButton } from '../PlaidLinkButton';
import toast from 'react-hot-toast';

// Mock dependencies
jest.mock('axios');
jest.mock('react-hot-toast');
jest.mock('react-plaid-link', () => ({
  usePlaidLink: jest.fn(),
}));

const mockedAxios = axios as jest.Mocked<typeof axios>;
const { usePlaidLink } = require('react-plaid-link');

describe('PlaidLinkButton', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Default mock for usePlaidLink
    usePlaidLink.mockReturnValue({
      open: jest.fn(),
      ready: false,
    });
  });

  describe('Component Rendering', () => {
    it('should render connect button with default text', () => {
      render(<PlaidLinkButton />);

      expect(screen.getByRole('button')).toBeInTheDocument();
      expect(screen.getByText('Connect Bank Account')).toBeInTheDocument();
    });

    it('should render with custom className', () => {
      const { container } = render(<PlaidLinkButton className="custom-class" />);

      const button = container.querySelector('.custom-class');
      expect(button).toBeInTheDocument();
    });

    it('should render disabled button when disabled prop is true', () => {
      render(<PlaidLinkButton disabled={true} />);

      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('should display bank icon when not loading', () => {
      const { container } = render(<PlaidLinkButton />);

      const icon = container.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });
  });

  describe('Link Token Creation', () => {
    it('should create link token on button click', async () => {
      const mockLinkToken = 'link-sandbox-test-token-123';
      mockedAxios.post.mockResolvedValueOnce({
        data: {
          link_token: mockLinkToken,
          expiration: '2025-01-10T12:00:00Z',
          request_id: 'req_123',
        },
      });

      render(<PlaidLinkButton userId="test_user_123" />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith(
          '/api/plaid/create_link_token',
          expect.objectContaining({
            user_id: 'test_user_123',
          })
        );
      });

      expect(toast.success).toHaveBeenCalledWith('Ready to connect bank account');
    });

    it('should use default user_id if not provided', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { link_token: 'token', expiration: '2025-01-10T12:00:00Z' },
      });

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith(
          '/api/plaid/create_link_token',
          expect.objectContaining({
            user_id: expect.stringMatching(/^user_\d+$/),
          })
        );
      });
    });

    it('should show error toast when link token creation fails', async () => {
      const errorMessage = 'Failed to initialize Plaid';
      mockedAxios.post.mockRejectedValueOnce({
        response: {
          data: { error: errorMessage },
        },
        message: errorMessage,
      });

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('Failed to initialize Plaid')
        );
      });
    });

    it('should show loading state during token creation', async () => {
      mockedAxios.post.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ data: { link_token: 'token' } }), 100))
      );

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      // Should show loading text
      await waitFor(() => {
        expect(screen.getByText('Initializing...')).toBeInTheDocument();
      });
    });
  });

  describe('Plaid Link Opening', () => {
    it('should open Plaid Link when token is ready', async () => {
      const mockOpen = jest.fn();
      usePlaidLink.mockReturnValue({
        open: mockOpen,
        ready: true,
      });

      mockedAxios.post.mockResolvedValueOnce({
        data: { link_token: 'link-token-123' },
      });

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOpen).toHaveBeenCalled();
      });
    });

    it('should not open Plaid Link if not ready', async () => {
      const mockOpen = jest.fn();
      usePlaidLink.mockReturnValue({
        open: mockOpen,
        ready: false,
      });

      mockedAxios.post.mockResolvedValueOnce({
        data: { link_token: 'link-token-123' },
      });

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalled();
      });

      // Should not call open immediately if not ready
      expect(mockOpen).not.toHaveBeenCalled();
    });
  });

  describe('Public Token Exchange', () => {
    it('should exchange public token on success', async () => {
      const mockOnSuccess = jest.fn();
      const publicToken = 'public-sandbox-test-token';
      const metadata = { institution: { name: 'Chase', institution_id: 'ins_123' } };

      mockedAxios.post
        .mockResolvedValueOnce({ data: { link_token: 'link-token' } })
        .mockResolvedValueOnce({
          data: { access_token: 'access-token-123', item_id: 'item_123' },
        });

      usePlaidLink.mockImplementation(({ onSuccess }) => {
        // Simulate Plaid Link success callback
        setTimeout(() => onSuccess(publicToken, metadata), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton onSuccess={mockOnSuccess} userId="test_user" />);

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledWith(
          '/api/plaid/exchange_public_token',
          expect.objectContaining({
            public_token: publicToken,
            user_id: 'test_user',
            metadata,
          })
        );
      });

      expect(toast.success).toHaveBeenCalledWith('Bank account connected successfully!');
      expect(mockOnSuccess).toHaveBeenCalledWith(publicToken, metadata);
    });

    it('should show error toast when token exchange fails', async () => {
      const publicToken = 'public-token';
      const metadata = { institution: { name: 'Chase' } };

      mockedAxios.post
        .mockResolvedValueOnce({ data: { link_token: 'link-token' } })
        .mockRejectedValueOnce({
          response: { data: { error: 'Invalid token' } },
          message: 'Exchange failed',
        });

      usePlaidLink.mockImplementation(({ onSuccess }) => {
        setTimeout(() => onSuccess(publicToken, metadata), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton />);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('Failed to connect bank')
        );
      });
    });

    it('should show exchanging state during token exchange', async () => {
      const publicToken = 'public-token';
      const metadata = { institution: { name: 'Chase' } };

      mockedAxios.post
        .mockResolvedValueOnce({ data: { link_token: 'link-token' } })
        .mockImplementation(
          () => new Promise((resolve) => setTimeout(() => resolve({ data: { access_token: 'token' } }), 100))
        );

      usePlaidLink.mockImplementation(({ onSuccess }) => {
        setTimeout(() => onSuccess(publicToken, metadata), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton />);

      await waitFor(() => {
        expect(screen.getByText('Connecting...')).toBeInTheDocument();
      });
    });
  });

  describe('Plaid Link Exit', () => {
    it('should handle user exit gracefully', async () => {
      const mockOnExit = jest.fn();
      const error = { error_code: 'USER_EXIT', error_message: 'User exited' };
      const metadata = {};

      usePlaidLink.mockImplementation(({ onExit }) => {
        setTimeout(() => onExit(error, metadata), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton onExit={mockOnExit} />);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('Plaid connection cancelled')
        );
      });

      expect(mockOnExit).toHaveBeenCalledWith(error, metadata);
    });

    it('should reset link token on exit to allow retry', async () => {
      mockedAxios.post.mockResolvedValue({ data: { link_token: 'link-token' } });

      usePlaidLink.mockImplementation(({ onExit }) => {
        setTimeout(() => onExit(null, {}), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton />);

      await waitFor(() => {
        expect(usePlaidLink).toHaveBeenCalled();
      });

      // Click button again should create new link token
      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockedAxios.post).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Button States', () => {
    it('should disable button when loading', async () => {
      mockedAxios.post.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ data: { link_token: 'token' } }), 100))
      );

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(button).toBeDisabled();
      });
    });

    it('should disable button when exchanging token', async () => {
      const publicToken = 'public-token';
      const metadata = {};

      mockedAxios.post
        .mockResolvedValueOnce({ data: { link_token: 'link-token' } })
        .mockImplementation(
          () => new Promise((resolve) => setTimeout(() => resolve({ data: { access_token: 'token' } }), 100))
        );

      usePlaidLink.mockImplementation(({ onSuccess }) => {
        setTimeout(() => onSuccess(publicToken, metadata), 0);
        return { open: jest.fn(), ready: true };
      });

      render(<PlaidLinkButton />);

      await waitFor(() => {
        const button = screen.getByRole('button');
        expect(button).toBeDisabled();
      });
    });

    it('should not allow click when disabled', () => {
      render(<PlaidLinkButton disabled={true} />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(mockedAxios.post).not.toHaveBeenCalled();
    });
  });

  describe('Error Handling', () => {
    it('should handle no link_token in response', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { expiration: '2025-01-10T12:00:00Z' }, // Missing link_token
      });

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('No link token received')
        );
      });
    });

    it('should handle network errors gracefully', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network error'));

      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          expect.stringContaining('Network error')
        );
      });
    });
  });

  describe('Accessibility', () => {
    it('should have accessible button role', () => {
      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      expect(button).toBeInTheDocument();
    });

    it('should support keyboard navigation', () => {
      render(<PlaidLinkButton />);

      const button = screen.getByRole('button');
      button.focus();

      expect(document.activeElement).toBe(button);
    });

    it('should have proper ARIA attributes when disabled', () => {
      render(<PlaidLinkButton disabled={true} />);

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('disabled');
    });
  });
});

describe('PlaidLinkButtonCompact', () => {
  it('should render compact variant with smaller padding', () => {
    const { PlaidLinkButtonCompact } = require('../PlaidLinkButton');
    const { container } = render(<PlaidLinkButtonCompact />);

    const button = container.querySelector('button');
    expect(button).toHaveClass('px-4', 'py-2', 'text-sm');
  });
});
