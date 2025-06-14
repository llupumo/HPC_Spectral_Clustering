from datetime import datetime, timedelta

def days_since_to_date(days_since):
    """
    Convert days since 1900-01-01 to a date.
    Args:
        days_since (int): Number of days since 1900-01-01.
    Returns:
        datetime.date: The corresponding date.
    """
    # Reference date: 1900-01-01
    start_date = datetime(1900, 1, 1)
    
    # Add the number of days to the start date
    result_date = start_date + timedelta(days=days_since)
    
    return result_date.date()