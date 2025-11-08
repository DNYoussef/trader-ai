import React, { useState, useCallback } from 'react';
import { usePlaidLink, PlaidLinkOnSuccess, PlaidLinkError } from 'react-plaid-link';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import dashboardAPI from '@/services/api';
import { exchangePublicToken as exchangeToken } from '@/services/authService';
import toast from 'react-hot-toast';

interface PlaidLinkButtonProps {
  onSuccess?: (publicToken: string, metadata: any) => void;
  onExit?: (error: PlaidLinkError | null, metadata: any) => void;
  className?: string;
  userId?: string;
  disabled?: boolean;
}

export const PlaidLinkButton: React.FC<PlaidLinkButtonProps> = ({
  onSuccess,
  onExit,
  className,
  userId,
  disabled = false,
}) => {
  const [linkToken, setLinkToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [exchanging, setExchanging] = useState(false);

  // Generate link token from backend
  const generateLinkToken = useCallback(async () => {
    if (loading || linkToken) return;

    setLoading(true);
    try {
      const response = await dashboardAPI.axiosClient.post('/api/plaid/create_link_token', {
        user_id: userId || `user_${Date.now()}`,
      });

      if (response.data.link_token) {
        setLinkToken(response.data.link_token);
        toast.success('Ready to connect bank account');
      } else {
        throw new Error('No link token received');
      }
    } catch (error: any) {
      console.error('Failed to create link token:', error);
      toast.error('Failed to initialize Plaid: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  }, [userId, loading, linkToken]);

  // Exchange public token for JWT token
  const handleExchangePublicToken = useCallback(async (publicToken: string, metadata: any) => {
    setExchanging(true);
    try {
      // Exchange public token for JWT token (stores in authService)
      const jwtToken = await exchangeToken(publicToken);

      toast.success('Bank account connected successfully!');
      console.log('JWT token received and stored');

      // Call success callback with metadata
      onSuccess?.(publicToken, metadata);
    } catch (error: any) {
      console.error('Failed to exchange public token:', error);
      toast.error('Failed to connect bank: ' + (error.response?.data?.error || error.message));
    } finally {
      setExchanging(false);
      // Reset link token to allow reconnection
      setLinkToken(null);
    }
  }, [onSuccess]);

  // Plaid Link handlers
  const handleOnSuccess: PlaidLinkOnSuccess = useCallback(
    (publicToken, metadata) => {
      console.log('Plaid Link success:', metadata);
      handleExchangePublicToken(publicToken, metadata);
    },
    [handleExchangePublicToken]
  );

  const handleOnExit = useCallback(
    (error: PlaidLinkError | null, metadata: any) => {
      console.log('Plaid Link exit:', error, metadata);
      if (error) {
        toast.error(`Plaid connection cancelled: ${error.error_message}`);
      }
      onExit?.(error, metadata);
      // Reset link token to allow retry
      setLinkToken(null);
    },
    [onExit]
  );

  // Initialize Plaid Link
  const { open, ready } = usePlaidLink({
    token: linkToken,
    onSuccess: handleOnSuccess,
    onExit: handleOnExit,
  });

  // Handle button click
  const handleClick = useCallback(() => {
    if (exchanging || disabled) return;

    if (linkToken && ready) {
      // Open Plaid Link if token exists
      open();
    } else {
      // Generate token first, then open
      generateLinkToken().then(() => {
        // Will open on next render when ready
      });
    }
  }, [linkToken, ready, open, generateLinkToken, exchanging, disabled]);

  // Auto-open when token becomes ready
  React.useEffect(() => {
    if (linkToken && ready && !exchanging) {
      open();
    }
  }, [linkToken, ready, open, exchanging]);

  const isLoading = loading || exchanging;
  const isDisabled = disabled || isLoading;

  return (
    <motion.button
      whileHover={!isDisabled ? { scale: 1.02 } : undefined}
      whileTap={!isDisabled ? { scale: 0.98 } : undefined}
      onClick={handleClick}
      disabled={isDisabled}
      className={clsx(
        'relative px-6 py-3 rounded-lg font-medium transition-all duration-200',
        'flex items-center justify-center space-x-2',
        'focus:outline-none focus:ring-2 focus:ring-offset-2',
        isDisabled
          ? 'bg-gray-300 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
          : 'bg-primary-600 hover:bg-primary-700 text-white shadow-md hover:shadow-lg focus:ring-primary-500',
        className
      )}
    >
      {/* Loading spinner */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-primary-600 rounded-lg">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
        </div>
      )}

      {/* Bank icon */}
      {!isLoading && (
        <svg
          className="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
          />
        </svg>
      )}

      {/* Button text */}
      <span>
        {loading
          ? 'Initializing...'
          : exchanging
          ? 'Connecting...'
          : linkToken && ready
          ? 'Opening Plaid...'
          : 'Connect Bank Account'}
      </span>
    </motion.button>
  );
};

// Compact variant for inline use
export const PlaidLinkButtonCompact: React.FC<PlaidLinkButtonProps> = (props) => {
  return (
    <PlaidLinkButton
      {...props}
      className={clsx(
        'px-4 py-2 text-sm',
        props.className
      )}
    />
  );
};

export default PlaidLinkButton;
