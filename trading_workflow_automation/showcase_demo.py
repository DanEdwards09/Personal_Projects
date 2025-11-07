import requests
import time
import json

BASE_URL = "http://localhost:5001"

def create_trade(trade_data):
    """Create a new trade and return the response"""
    response = requests.post(
        f"{BASE_URL}/trades", 
        json=trade_data
    )
    return response.json()

def get_trade(trade_id):
    """Get details of a specific trade"""
    response = requests.get(f"{BASE_URL}/trades/{trade_id}")
    return response.json()

def run_showcase():
    print("\n===== TRADE SETTLEMENT DASHBOARD SHOWCASE =====\n")
    
    # Successful Trade
    print("Creating a successful trade...")
    successful_trade = create_trade({
        "security_id": "AAPL",
        "quantity": 100,
        "price": 175.25,
        "buyer": "fund_alpha",
        "seller": "broker_xyz",
        "clearing_member": "GLOBAL_CLEARING_HOUSE"
    })
    print(f"  Result: {successful_trade['message']}")
    
    time.sleep(2)  # Give time for processing

    # Trade with validation failure
    print("\nCreating a trade that will fail validation (negative quantity)...")
    validation_failed_trade = create_trade({
        "security_id": "MSFT",
        "quantity": -50,
        "price": 325.75,
        "buyer": "pension_fund",
        "seller": "hedge_fund",
        "clearing_member": "EUROCLEAR"
    })
    print(f"  Result: {validation_failed_trade['message']}")
    
    time.sleep(2)
    
    # Trade with clearing failure
    print("\nCreating a high-value trade that will fail clearing...")
    clearing_failed_trade = create_trade({
        "security_id": "GOOGL",
        "quantity": 10000,
        "price": 142.80,
        "buyer": "mutual_fund",
        "seller": "proprietary_trader",
        "clearing_member": "DTCC"
    })
    print(f"  Result: {clearing_failed_trade['message']}")
    
    time.sleep(2)
    
    # Trade with restricted security
    print("\nCreating a trade with a restricted security...")
    restricted_trade = create_trade({
        "security_id": "RESTRICTED_XYZ",
        "quantity": 200,
        "price": 50.75,
        "buyer": "hedge_fund_b",
        "seller": "bank_a",
        "clearing_member": "ICE_CLEAR"
    })
    print(f"  Result: {restricted_trade['message']}")
    
    print("\n===== DEMONSTRATION COMPLETED =====")
    print("\nPlease check the dashboard to see all trades and their statuses.")
    print(f"Dashboard URL: {BASE_URL}/dashboard")

if __name__ == "__main__":
    run_showcase()