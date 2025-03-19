"""
Simple Trade Settlement Workflow

A lightweight implementation of a trade settlement process with basic workflow steps.
"""

import sqlite3
import json
from datetime import datetime, timedelta
import logging
from enum import Enum

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleTradeSys")

class TradeStatus(Enum):
    """Simple enum for trade status tracking"""
    NEW = "NEW"
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    MATCHED = "MATCHED"
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class SimpleTradeWorkflow:
    """A simplified trade settlement workflow system"""
    
    def __init__(self, db_path="simple_trades.db"):
        """Initialize the workflow with a database"""
        self.db_path = db_path
        self._setup_database()
        logger.info("Simple Trade Settlement Workflow initialized")
    
    def _setup_database(self):
        """Set up a basic SQLite database for trade storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create a simple trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            security_id TEXT,
            quantity REAL,
            price REAL,
            buyer TEXT,
            seller TEXT,
            trade_date TEXT,
            settlement_date TEXT,
            status TEXT,
            notes TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        ''')
        
        # Create a simple audit log table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT,
            action TEXT,
            old_status TEXT,
            new_status TEXT,
            timestamp TEXT,
            user TEXT,
            notes TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup complete")
    
    def create_trade(self, trade_data):
        """Create a new trade record"""
        # Generate a simple trade ID if not provided
        trade_id = trade_data.get('trade_id', f"T{int(datetime.now().timestamp())}")
        
        # Set default dates if not provided
        trade_date = trade_data.get('trade_date', datetime.now().strftime('%Y-%m-%d'))
        settlement_date = trade_data.get('settlement_date')
        
        if not settlement_date:
            # Default T+2 settlement
            settlement_date = (datetime.strptime(trade_date, '%Y-%m-%d') + timedelta(days=2)).strftime('%Y-%m-%d')
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert the new trade with NEW status
            cursor.execute('''
            INSERT INTO trades (
                trade_id, security_id, quantity, price, buyer, seller,
                trade_date, settlement_date, status, notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trade_data.get('security_id'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('buyer'),
                trade_data.get('seller'),
                trade_date,
                settlement_date,
                TradeStatus.NEW.value,
                None,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            # Log the action
            self._log_action(
                cursor, 
                trade_id, 
                "CREATE", 
                None, 
                TradeStatus.NEW.value, 
                "Trade created"
            )
            
            # Commit transaction
            conn.commit()
            logger.info(f"Created trade {trade_id}")
            
            # Proceed to validation automatically
            conn.close()
            self.validate_trade(trade_id)
            
            return {
                "success": True,
                "trade_id": trade_id,
                "status": TradeStatus.NEW.value,
                "message": "Trade created successfully"
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating trade: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create trade: {str(e)}"
            }
        finally:
            if conn:
                conn.close()
    
    def validate_trade(self, trade_id):
        """Validate a trade for correctness"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the trade data
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                logger.warning(f"Trade {trade_id} not found for validation")
                return {
                    "success": False,
                    "message": "Trade not found"
                }
            
            # Convert to dict for easier access
            columns = [col[0] for col in cursor.description]
            trade_data = dict(zip(columns, trade))
            
            # Simple validation rules
            validation_errors = []
            
            # Check for required fields
            if not trade_data.get('security_id'):
                validation_errors.append("Security ID is required")
            
            # Check quantity and price are positive
            if trade_data.get('quantity', 0) <= 0:
                validation_errors.append("Quantity must be positive")
            
            if trade_data.get('price', 0) <= 0:
                validation_errors.append("Price must be positive")
            
            # Check dates are valid
            trade_date = datetime.strptime(trade_data.get('trade_date'), '%Y-%m-%d')
            settlement_date = datetime.strptime(trade_data.get('settlement_date'), '%Y-%m-%d')
            
            if settlement_date < trade_date:
                validation_errors.append("Settlement date must be on or after trade date")
            
            # Update trade status based on validation
            if validation_errors:
                # Validation failed
                new_status = TradeStatus.VALIDATION_FAILED.value
                notes = "; ".join(validation_errors)
                logger.warning(f"Trade {trade_id} validation failed: {notes}")
            else:
                # Validation successful
                new_status = TradeStatus.VALIDATED.value
                notes = "Trade validated successfully"
                logger.info(f"Trade {trade_id} validated successfully")
            
            # Update the trade record
            cursor.execute('''
            UPDATE trades 
            SET status = ?, notes = ?, updated_at = ?
            WHERE trade_id = ?
            ''', (
                new_status,
                notes,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade_id
            ))
            
            # Log the action
            self._log_action(
                cursor, 
                trade_id, 
                "VALIDATE", 
                TradeStatus.NEW.value, 
                new_status, 
                notes
            )
            
            # Commit transaction
            conn.commit()
            
            # Automatically proceed to matching if validation passed
            if new_status == TradeStatus.VALIDATED.value:
                conn.close()
                self.match_trade(trade_id)
            
            return {
                "success": True,
                "status": new_status,
                "message": notes
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error validating trade {trade_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error validating trade: {str(e)}"
            }
        finally:
            if conn:
                conn.close()
    
    def match_trade(self, trade_id):
        """Match the trade with counterparty (simplified simulation)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the trade data
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                logger.warning(f"Trade {trade_id} not found for matching")
                return {
                    "success": False,
                    "message": "Trade not found"
                }
            
            # Convert to dict for easier access
            columns = [col[0] for col in cursor.description]
            trade_data = dict(zip(columns, trade))
            
            # In a real system, this would involve confirming details with the counterparty
            # For simplicity, we'll just simulate a 90% match success rate
            import random
            match_successful = random.random() < 0.9
            
            if match_successful:
                new_status = TradeStatus.MATCHED.value
                notes = "Trade matched successfully with counterparty"
                logger.info(f"Trade {trade_id} matched successfully")
                
                # Proceed to settlement
                settlement_date = trade_data.get('settlement_date')
                today = datetime.now().strftime('%Y-%m-%d')
                
                # If settlement date is today or in the past, settle immediately
                if settlement_date <= today:
                    settlement_now = True
                else:
                    # In a real system, this would be scheduled for the settlement date
                    # For simplicity, we'll settle immediately regardless
                    settlement_now = True
            else:
                new_status = TradeStatus.FAILED.value
                notes = "Trade matching failed: counterparty rejected trade"
                logger.warning(f"Trade {trade_id} matching failed")
                settlement_now = False
            
            # Update the trade record
            cursor.execute('''
            UPDATE trades 
            SET status = ?, notes = ?, updated_at = ?
            WHERE trade_id = ?
            ''', (
                new_status,
                notes,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade_id
            ))
            
            # Log the action
            self._log_action(
                cursor, 
                trade_id, 
                "MATCH", 
                TradeStatus.VALIDATED.value, 
                new_status, 
                notes
            )
            
            # Commit transaction
            conn.commit()
            
            # Proceed to settlement if matched
            if settlement_now:
                conn.close()
                self.settle_trade(trade_id)
            
            return {
                "success": True,
                "status": new_status,
                "message": notes
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error matching trade {trade_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error matching trade: {str(e)}"
            }
        finally:
            if conn:
                conn.close()
    
    def settle_trade(self, trade_id):
        """Complete the final settlement of the trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the trade data
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                logger.warning(f"Trade {trade_id} not found for settlement")
                return {
                    "success": False,
                    "message": "Trade not found"
                }
            
            # Convert to dict for easier access
            columns = [col[0] for col in cursor.description]
            trade_data = dict(zip(columns, trade))
            
            # In a real system, this would involve actual fund and security transfers
            # For simplicity, we'll just simulate a 95% settlement success rate
            import random
            settlement_successful = random.random() < 0.95
            
            if settlement_successful:
                new_status = TradeStatus.SETTLED.value
                notes = "Trade settled successfully"
                logger.info(f"Trade {trade_id} settled successfully")
            else:
                new_status = TradeStatus.FAILED.value
                notes = "Settlement failed: insufficient funds or securities"
                logger.warning(f"Trade {trade_id} settlement failed")
            
            # Update the trade record
            cursor.execute('''
            UPDATE trades 
            SET status = ?, notes = ?, updated_at = ?
            WHERE trade_id = ?
            ''', (
                new_status,
                notes,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade_id
            ))
            
            # Log the action
            self._log_action(
                cursor, 
                trade_id, 
                "SETTLE", 
                TradeStatus.MATCHED.value, 
                new_status, 
                notes
            )
            
            # Commit transaction
            conn.commit()
            
            return {
                "success": True,
                "status": new_status,
                "message": notes
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error settling trade {trade_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error settling trade: {str(e)}"
            }
        finally:
            if conn:
                conn.close()
    
    def cancel_trade(self, trade_id, reason):
        """Cancel a trade at any point in the workflow"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get the trade data
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                logger.warning(f"Trade {trade_id} not found for cancellation")
                return {
                    "success": False,
                    "message": "Trade not found"
                }
            
            # Convert to dict for easier access
            columns = [col[0] for col in cursor.description]
            trade_data = dict(zip(columns, trade))
            
            # Can't cancel an already settled trade
            if trade_data.get('status') == TradeStatus.SETTLED.value:
                logger.warning(f"Cannot cancel settled trade {trade_id}")
                return {
                    "success": False,
                    "message": "Cannot cancel a trade that has already settled"
                }
            
            # Update the trade record
            old_status = trade_data.get('status')
            new_status = TradeStatus.CANCELLED.value
            
            cursor.execute('''
            UPDATE trades 
            SET status = ?, notes = ?, updated_at = ?
            WHERE trade_id = ?
            ''', (
                new_status,
                f"Trade cancelled: {reason}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                trade_id
            ))
            
            # Log the action
            self._log_action(
                cursor, 
                trade_id, 
                "CANCEL", 
                old_status, 
                new_status, 
                f"Trade cancelled: {reason}"
            )
            
            # Commit transaction
            conn.commit()
            logger.info(f"Trade {trade_id} cancelled: {reason}")
            
            return {
                "success": True,
                "status": new_status,
                "message": f"Trade cancelled: {reason}"
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error cancelling trade {trade_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error cancelling trade: {str(e)}"
            }
        finally:
            if conn:
                conn.close()
    
    def get_trade(self, trade_id):
        """Retrieve a specific trade by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
            trade = cursor.fetchone()
            
            if not trade:
                return None
            
            return dict(trade)
        except Exception as e:
            logger.error(f"Error retrieving trade {trade_id}: {str(e)}")
            return None
        finally:
            conn.close()
    
    def get_trade_history(self, trade_id):
        """Get the audit history for a specific trade"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            SELECT * FROM audit_log 
            WHERE trade_id = ? 
            ORDER BY timestamp ASC
            """, (trade_id,))
            
            history = cursor.fetchall()
            return [dict(row) for row in history]
        except Exception as e:
            logger.error(f"Error retrieving history for trade {trade_id}: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_trades_by_status(self, status):
        """Get all trades with a specific status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM trades WHERE status = ?", (status,))
            trades = cursor.fetchall()
            return [dict(trade) for trade in trades]
        except Exception as e:
            logger.error(f"Error retrieving trades with status {status}: {str(e)}")
            return []
        finally:
            conn.close()
    
    def get_all_trades(self):
        """Get all trades in the system"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM trades ORDER BY created_at DESC")
            trades = cursor.fetchall()
            return [dict(trade) for trade in trades]
        except Exception as e:
            logger.error(f"Error retrieving all trades: {str(e)}")
            return []
        finally:
            conn.close()
    
    def _log_action(self, cursor, trade_id, action, old_status, new_status, notes, user="system"):
        """Internal method to log actions to the audit table"""
        cursor.execute('''
        INSERT INTO audit_log (
            trade_id, action, old_status, new_status, timestamp, user, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id,
            action,
            old_status,
            new_status,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user,
            notes
        ))


# Simple CLI interface for testing
if __name__ == "__main__":
    import sys
    
    print("\n===== Simple Trade Settlement Workflow =====\n")
    
    workflow = SimpleTradeWorkflow()
    
    # Create a sample trade
    sample_trade = {
        "security_id": "AAPL",
        "quantity": 100,
        "price": 150.50,
        "buyer": "client_a",
        "seller": "broker_x"
    }
    
    print(f"Creating sample trade: {json.dumps(sample_trade, indent=2)}")
    result = workflow.create_trade(sample_trade)
    
    if result["success"]:
        trade_id = result["trade_id"]
        print(f"Trade created with ID: {trade_id}")
        
        # Get the final trade details
        trade = workflow.get_trade(trade_id)
        print(f"\nFinal trade status: {trade['status']}")
        print(f"Notes: {trade['notes']}")
        
        # Show the trade history
        print("\nTrade history:")
        history = workflow.get_trade_history(trade_id)
        for entry in history:
            print(f"  {entry['timestamp']} - {entry['action']}: {entry.get('old_status') or 'None'} → {entry['new_status']}")
            if entry['notes']:
                print(f"    Notes: {entry['notes']}")
    else:
        print(f"Failed to create trade: {result['message']}")