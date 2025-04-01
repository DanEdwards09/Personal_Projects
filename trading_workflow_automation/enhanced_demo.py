"""
Demo script to test the Enhanced Trade Settlement Workflow with Clearing

This demo shows all stages of a trade lifecycle including the clearing process.
"""

from enhanced_trade_workflow import EnhancedTradeWorkflow, TradeStatus
import json
import time
from datetime import datetime, timedelta

def run_enhanced_demo():
    """Run a demonstration of the enhanced trade settlement workflow with clearing"""
    print("\n===== ENHANCED TRADE SETTLEMENT WORKFLOW WITH CLEARING DEMO =====\n")
    
    # Create workflow instance
    workflow = EnhancedTradeWorkflow("demo_enhanced_trades.db")
    print("Enhanced Trade Settlement Workflow initialized")
    
    # Create sample trades
    print("\nCreating sample trades...\n")
    
    trades = [
        {
            "security_id": "AAPL",
            "quantity": 100,
            "price": 175.25,
            "buyer": "fund_alpha",
            "seller": "broker_xyz",
            "clearing_member": "GLOBAL_CLEARING_HOUSE"
        },
        {
            "security_id": "MSFT",
            "quantity": 50,
            "price": 325.75,
            "buyer": "pension_fund",
            "seller": "hedge_fund",
            "clearing_member": "EUROCLEAR"
        },
        {
            # This trade will fail validation (negative quantity)
            "security_id": "AMZN",
            "quantity": -25,
            "price": 140.50,
            "buyer": "retail_investor",
            "seller": "market_maker",
            "clearing_member": "CLEARSTREAM"
        },
        {
            # This trade will fail clearing (>$1M value)
            "security_id": "GOOGL",
            "quantity": 10000,
            "price": 142.80,
            "buyer": "mutual_fund",
            "seller": "proprietary_trader",
            "clearing_member": "DTCC"
        },
        {
            # This trade will pass validation but fail clearing (restricted security)
            "security_id": "RESTRICTED_XYZ",
            "quantity": 200,
            "price": 50.75,
            "buyer": "hedge_fund_b",
            "seller": "bank_a",
            "clearing_member": "ICE_CLEAR"
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
            print(f"    Notes: {trade['notes']}")
            if trade['status'] in [TradeStatus.CLEARED.value, TradeStatus.MATCHED.value, TradeStatus.SETTLED.value]:
                print(f"    Clearing Member: {trade['clearing_member']}")
                print(f"    Initial Margin: ${float(trade['initial_margin']):.2f}")
                print(f"    Clearing Fees: ${float(trade['clearing_fees']):.2f}")
                print(f"    Net Settlement Amount: ${float(trade['net_settlement_amount']):.2f}")