"""
Demo script to test the Simple Trade Settlement Workflow
"""

from simple_trade_workflow import SimpleTradeWorkflow, TradeStatus
import json
import time
from datetime import datetime, timedelta

def run_demo():
    """Run a demonstration of the simple trade settlement workflow"""
    print("\n===== SIMPLE TRADE SETTLEMENT WORKFLOW DEMO =====\n")
    
    # Create workflow instance
    workflow = SimpleTradeWorkflow("demo_trades.db")
    print("Trade Settlement Workflow initialized")
    
    # Create sample trades
    print("\nCreating sample trades...\n")
    
    trades = [
        {
            "security_id": "AAPL",
            "quantity": 100,
            "price": 175.25,
            "buyer": "fund_alpha",
            "seller": "broker_xyz"
        },
        {
            "security_id": "MSFT",
            "quantity": 50,
            "price": 325.75,
            "buyer": "pension_fund",
            "seller": "hedge_fund"
        },
        {
            # This trade will fail validation (negative quantity)
            "security_id": "AMZN",
            "quantity": -25,
            "price": 140.50,
            "buyer": "retail_investor",
            "seller": "market_maker"
        },
        {
            "security_id": "GOOGL",
            "quantity": 30,
            "price": 142.80,
            "buyer": "mutual_fund",
            "seller": "proprietary_trader",
            # Settlement date in the past will fail validation
            "trade_date": datetime.now().strftime('%Y-%m-%d'),
            "settlement_date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        }
    ]
    
    # Initiate trades
    trade_ids = []
    for i, trade_data in enumerate(trades):
        print(f"Creating Trade {i+1}: {trade_data['security_id']}")
        result = workflow.create_trade(trade_data)
        print(f"  Result: {result['message']}")
        
        if result["success"]:
            trade_ids.append(result["trade_id"])
    
    # Give some time for workflow processing
    print("\nProcessing trades...\n")
    time.sleep(1)
    
    # Show current status of all trades
    print("Current status of all trades:")
    for trade_id in trade_ids:
        trade = workflow.get_trade(trade_id)
        if trade:
            print(f"  Trade {trade_id} ({trade['security_id']}): {trade['status']}")
            if trade['notes']:
                print(f"    Notes: {trade['notes']}")
    
    # Cancel one trade
    if trade_ids:
        cancel_trade_id = trade_ids[0]
        print(f"\nCancelling trade {cancel_trade_id}...")
        result = workflow.cancel_trade(cancel_trade_id, "Client requested cancellation")
        print(f"  Result: {result['message']}")
    
    # Print final status summary
    print("\nFinal status counts:")
    for status in TradeStatus:
        trades = workflow.get_trades_by_status(status.value)
        print(f"  {status.value}: {len(trades)} trades")
    
    # Show detailed history for one trade
    if len(trade_ids) > 1:
        history_trade_id = trade_ids[1]
        print(f"\nDetailed history for trade {history_trade_id}:")
        
        trade = workflow.get_trade(history_trade_id)
        print(f"  Security: {trade['security_id']}")
        print(f"  Quantity: {trade['quantity']}")
        print(f"  Price: ${trade['price']}")
        print(f"  Total Value: ${trade['quantity'] * trade['price']}")
        print(f"  Current Status: {trade['status']}")
        
        history = workflow.get_trade_history(history_trade_id)
        print("\n  Workflow History:")
        for entry in history:
            print(f"    {entry['timestamp']} - {entry['action']}: {entry.get('old_status') or 'None'} → {entry['new_status']}")
    
    print("\n===== DEMONSTRATION COMPLETED =====\n")

if __name__ == "__main__":
    run_demo()