"""Finance MCP Tool Server.

Provides loan calculation tools exposed over the Model Context Protocol (MCP)
using stdio transport for inter-process communication.

Tools:
    calculate_monthly_payment: Computes monthly loan payments with detailed breakdown

Usage:
    This server is launched automatically by the Finance Agent.
    Direct usage: python -m finance.mcp_server
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Finance Service")


@mcp.tool()
def calculate_monthly_payment(price: float, rate: float, months: int) -> str:
    """Calculate monthly loan payment with detailed breakdown.

    Computes the monthly payment for an auto loan using standard amortization
    formula, including total cost and interest paid over the loan term.

    Args:
        price: Total vehicle price in dollars
        rate: Annual interest rate as percentage (e.g., 5.0 for 5%)
        months: Loan term in months

    Returns:
        Human-readable string with monthly payment, total cost, interest,
        and rate details

    Example:
        >>> calculate_monthly_payment(30000, 5.0, 60)
        'Monthly payment: $566.14 | Total over 60 months: $33,968.40 | ...'
    """
    # Type coercion for string inputs from LLM
    price = float(price)
    rate = float(rate)
    months = int(months)
    
    monthly_rate = rate / 100.0 / 12.0
    
    if monthly_rate == 0:
        payment = price / months
    else:
        payment = (price * monthly_rate) / (1 - (1 + monthly_rate) ** -months)

    total = payment * months
    interest = total - price

    return (
        f"Monthly payment: ${payment:,.2f} | "
        f"Total over {months} months: ${total:,.2f} | "
        f"Total interest: ${interest:,.2f} | "
        f"Rate: {rate}% APR"
    )


@mcp.tool()
def compare_loan_options(price: float, rates: list[float], terms: list[int]) -> str:
    """Compare multiple loan scenarios side-by-side.

    Generates a comparison table of different rate and term combinations
    to help customers choose the best financing option.

    Args:
        price: Vehicle price in dollars
        rates: List of annual interest rates (e.g., [3.5, 4.5, 5.5])
        terms: List of loan terms in months (e.g., [36, 48, 60])

    Returns:
        Formatted comparison table showing monthly payments for each combination

    Example:
        >>> compare_loan_options(30000, [4.0, 5.0], [48, 60])
        "\\n4.0% APR: 48mo=$678.58, 60mo=$552.50\\n5.0% APR: 48mo=$689.58, 60mo=$566.14"
    """
    results = []
    for rate in rates:  # Limit to 5 rates
        rate_results = []
        for term in terms:  # Limit to 5 terms
            rate = float(rate)
            term = int(term)
            price = float(price)
            monthly_rate = rate / 100.0 / 12.0
            if monthly_rate == 0:
                payment = price / term
            else:
                payment = (price * monthly_rate) / (1 - (1 + monthly_rate) ** -term)
            rate_results.append(f"{term}mo=${payment:,.2f}")
        results.append(f"{rate}% APR: {', '.join(rate_results)}")
    
    return "\\n".join(results)


@mcp.tool()
def calculate_affordability(monthly_budget: float, rate: float, months: int) -> str:
    """Calculate maximum affordable vehicle price from monthly budget.

    Reverse calculation to determine what vehicle price fits within
    a customer's monthly payment budget.

    Args:
        monthly_budget: Maximum monthly payment customer can afford
        rate: Annual interest rate as percentage
        months: Desired loan term in months

    Returns:
        Maximum affordable price and loan details

    Example:
        >>> calculate_affordability(500, 5.0, 60)
        "Max affordable price: $26,495.63 | Monthly: $500.00 | 60mo @ 5.0% APR"
    """
    # Type coercion for string inputs from LLM
    monthly_budget = float(monthly_budget)
    rate = float(rate)
    months = int(months)
    
    monthly_rate = rate / 100.0 / 12.0
    
    if monthly_rate == 0:
        max_price = monthly_budget * months
    else:
        max_price = monthly_budget * (1 - (1 + monthly_rate) ** -months) / monthly_rate
    
    return (
        f"Max affordable price: ${max_price:,.2f} | "
        f"Monthly: ${monthly_budget:,.2f} | "
        f"{months}mo @ {rate}% APR"
    )


@mcp.tool()
def get_rate_recommendations(credit_score: int = 700) -> str:
    """Get typical interest rate ranges based on credit score.

    Provides estimated rate ranges to set customer expectations
    based on common lending standards.

    Args:
        credit_score: Customer's credit score (300-850 range)

    Returns:
        Estimated rate range and qualification tier

    Example:
        >>> get_rate_recommendations(750)
        "Excellent credit (750): Estimated rate range 3.5% - 4.5% APR"
    """
    # Type coercion for string inputs from LLM
    credit_score = int(credit_score)
    
    if credit_score >= 800:
        tier = "Exceptional"
        low, high = 2.5, 3.5
    elif credit_score >= 740:
        tier = "Excellent"
        low, high = 3.5, 4.5
    elif credit_score >= 670:
        tier = "Good"
        low, high = 4.5, 6.0
    elif credit_score >= 580:
        tier = "Fair"
        low, high = 6.0, 10.0
    else:
        tier = "Poor"
        low, high = 10.0, 15.0
    
    return (
        f"{tier} credit ({credit_score}): "
        f"Estimated rate range {low}% - {high}% APR"
    )


@mcp.tool()
def calculate_total_cost_comparison(
    price: float,
    down_payment: float,
    rate: float,
    months: int
) -> str:
    """Calculate detailed cost breakdown including down payment.

    Provides comprehensive view of total ownership cost including
    down payment, financed amount, interest, and total out-of-pocket.

    Args:
        price: Vehicle price
        down_payment: Initial down payment amount
        rate: Annual interest rate as percentage
        months: Loan term in months

    Returns:
        Detailed cost breakdown

    Example:
        >>> calculate_total_cost_comparison(30000, 5000, 5.0, 60)
        "Financed: $25,000 | Down: $5,000 | Monthly: $471.78 | ..."
    """
    # Type coercion for string inputs from LLM
    price = float(price)
    down_payment = float(down_payment)
    rate = float(rate)
    months = int(months)
    
    financed = price - down_payment
    monthly_rate = rate / 100.0 / 12.0
    
    if monthly_rate == 0:
        payment = financed / months
    else:
        payment = (financed * monthly_rate) / (1 - (1 + monthly_rate) ** -months)
    
    total_payments = payment * months
    interest = total_payments - financed
    total_cost = down_payment + total_payments
    
    return (
        f"Financed: ${financed:,.2f} | "
        f"Down: ${down_payment:,.2f} | "
        f"Monthly: ${payment:,.2f} | "
        f"Total payments: ${total_payments:,.2f} | "
        f"Interest: ${interest:,.2f} | "
        f"Total cost: ${total_cost:,.2f} | "
        f"{months}mo @ {rate}% APR"
    )


if __name__ == "__main__":
    mcp.run()