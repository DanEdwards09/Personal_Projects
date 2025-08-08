"""
Enhanced REST API for the Trade Settlement Workflow with Clearing using Flask
"""

from flask import Flask, request, jsonify, render_template
from enhanced_trade_workflow import EnhancedTradeWorkflow, TradeStatus
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit

app = Flask(__name__)
workflow = EnhancedTradeWorkflow()

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Define the emit_trade_update function first
def emit_trade_update(trade_id, status, details=None):
    """Emit a trade update event to connected clients"""
    trade = workflow.get_trade(trade_id)
    if trade:
        socketio.emit('trade_update', {
            'trade_id': trade_id,
            'old_status': details.get('old_status') if details else None,
            'new_status': status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trade': trade
        })

# Define the event handler functions before registering them
def on_trade_validated(trade_id, old_status, new_status):
    """Callback when a trade is validated"""
    emit_trade_update(trade_id, new_status, {'old_status': old_status})

def on_trade_cleared(trade_id, old_status, new_status):
    """Callback when a trade is cleared"""
    emit_trade_update(trade_id, new_status, {'old_status': old_status})

def on_trade_matched(trade_id, old_status, new_status):
    """Callback when a trade is matched"""
    emit_trade_update(trade_id, new_status, {'old_status': old_status})

def on_trade_settled(trade_id, old_status, new_status):
    """Callback when a trade is settled"""
    emit_trade_update(trade_id, new_status, {'old_status': old_status})

def on_trade_failed(trade_id, old_status, new_status):
    """Callback when a trade fails"""
    emit_trade_update(trade_id, new_status, {'old_status': old_status})

# Now register the event handlers
workflow.on_status_change(TradeStatus.VALIDATED.value, on_trade_validated)
workflow.on_status_change(TradeStatus.CLEARED.value, on_trade_cleared)
workflow.on_status_change(TradeStatus.MATCHED.value, on_trade_matched)
workflow.on_status_change(TradeStatus.SETTLED.value, on_trade_settled)
workflow.on_status_change(TradeStatus.FAILED.value, on_trade_failed)

@app.route('/')
def index():
    """Homepage with links to API documentation"""
    return """
    <h1>Enhanced Trade Settlement API with Clearing</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/trades">/trades</a> - GET all trades or POST to create a new trade</li>
        <li><a href="/trades/status/NEW">/trades/status/{status}</a> - GET trades by status</li>
        <li>/trades/{trade_id} - GET details for a specific trade</li>
        <li>/trades/{trade_id}/history - GET history for a specific trade</li>
        <li>/trades/{trade_id}/cancel - POST to cancel a trade</li>
        <li>/trades/{trade_id}/clearing - GET clearing details for a trade</li>
        <li><a href="/dashboard">/dashboard</a> - Trade settlement dashboard</li>
    </ul>
    """

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard frontend"""
    return render_template('dashboard.html')

@app.route('/trades', methods=['GET', 'POST'])
def handle_trades():
    """Get all trades or create a new trade"""
    if request.method == 'GET':
        # Get all trades
        trades = workflow.get_all_trades()
        return jsonify({
            "status": "success",
            "count": len(trades),
            "trades": trades
        })
    elif request.method == 'POST':
        # Create a new trade
        trade_data = request.json
        result = workflow.create_trade(trade_data)
        
        if result["success"]:
            # Emit event for new trade
            emit_trade_update(result["trade_id"], TradeStatus.NEW.value)
            return jsonify(result), 201
        else:
            return jsonify(result), 400

@app.route('/trades/<trade_id>', methods=['GET'])
def get_trade(trade_id):
    """Get details for a specific trade"""
    trade = workflow.get_trade(trade_id)
    
    if trade:
        return jsonify({
            "status": "success",
            "trade": trade
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Trade not found"
        }), 404

@app.route('/trades/<trade_id>/history', methods=['GET'])
def get_trade_history(trade_id):
    """Get history for a specific trade"""
    # First check if the trade exists
    trade = workflow.get_trade(trade_id)
    
    if not trade:
        return jsonify({
            "status": "error",
            "message": "Trade not found"
        }), 404
    
    history = workflow.get_trade_history(trade_id)
    
    return jsonify({
        "status": "success",
        "trade_id": trade_id,
        "history": history
    })

@app.route('/trades/status/<status>', methods=['GET'])
def get_trades_by_status(status):
    """Get all trades with a specific status"""
    # Validate the status
    try:
        # Check if the provided status is valid
        valid_status = TradeStatus(status).value
    except ValueError:
        return jsonify({
            "status": "error",
            "message": f"Invalid status: {status}"
        }), 400
    
    trades = workflow.get_trades_by_status(status)
    
    return jsonify({
        "status": "success",
        "count": len(trades),
        "status_filter": status,
        "trades": trades
    })

@app.route('/trades/<trade_id>/cancel', methods=['POST'])
def cancel_trade(trade_id):
    """Cancel a specific trade"""
    # Get the cancellation reason from request
    data = request.json
    reason = data.get("reason", "No reason provided")
    
    # Get the current trade status before cancellation
    trade = workflow.get_trade(trade_id)
    if not trade:
        return jsonify({
            "status": "error",
            "message": "Trade not found"
        }), 404
    
    old_status = trade["status"]
    
    result = workflow.cancel_trade(trade_id, reason)
    
    if result["success"]:
        # Emit event for canceled trade
        emit_trade_update(trade_id, TradeStatus.CANCELLED.value, {'old_status': old_status})
        return jsonify(result)
    else:
        return jsonify(result), 400

@app.route('/trades/<trade_id>/clearing', methods=['GET'])
def get_trade_clearing_details(trade_id):
    """Get detailed clearing information for a trade"""
    trade = workflow.get_trade(trade_id)
    
    if not trade:
        return jsonify({
            "status": "error",
            "message": "Trade not found"
        }), 404
    
    # Check if trade has been through clearing
    if trade["status"] not in [TradeStatus.CLEARED.value, TradeStatus.MATCHED.value, TradeStatus.SETTLED.value]:
        return jsonify({
            "status": "error",
            "message": "Trade has not yet been cleared"
        }), 400
    
    # Extract clearing details
    clearing_details = {
        "trade_id": trade_id,
        "security_id": trade["security_id"],
        "quantity": trade["quantity"],
        "price": trade["price"],
        "trade_value": float(trade["quantity"]) * float(trade["price"]),
        "clearing_member": trade["clearing_member"],
        "initial_margin": trade["initial_margin"],
        "variation_margin": trade["variation_margin"],
        "clearing_fees": trade["clearing_fees"],
        "net_settlement_amount": trade["net_settlement_amount"],
        "status": trade["status"],
        "settlement_date": trade["settlement_date"]
    }
    
    return jsonify({
        "status": "success",
        "clearing_details": clearing_details
    })

# Dashboard API endpoints
@app.route('/api/dashboard/summary', methods=['GET'])
def dashboard_summary():
    """Get a summary of trade statuses for the dashboard"""
    summary = {}
    
    # Count trades by status
    for status in TradeStatus:
        trades = workflow.get_trades_by_status(status.value)
        summary[status.value] = len(trades)
    
    # Get recent failed or pending settlements
    failed_trades = workflow.get_trades_by_status(TradeStatus.FAILED.value)
    failed_trades = sorted(failed_trades, key=lambda x: x['updated_at'], reverse=True)[:5]
    
    return jsonify({
        "status": "success",
        "status_counts": summary,
        "recent_failed_trades": failed_trades,
        "total_trades": len(workflow.get_all_trades())
    })

@app.route('/api/dashboard/settlement-timeline', methods=['GET'])
def settlement_timeline():
    """Get settlement data for the next 5 days"""
    today = datetime.now().date()
    timeline = {}
    
    # Initialize timeline with next 5 days
    for i in range(5):
        date = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        timeline[date] = {"count": 0, "value": 0.0}
    
    # Get all trades
    all_trades = workflow.get_all_trades()
    
    # Populate timeline with pending settlements
    for trade in all_trades:
        if trade['status'] in [TradeStatus.VALIDATED.value, TradeStatus.CLEARED.value, TradeStatus.MATCHED.value]:
            settlement_date = trade.get('settlement_date')
            if settlement_date in timeline:
                timeline[settlement_date]["count"] += 1
                timeline[settlement_date]["value"] += float(trade['quantity']) * float(trade['price'])
    
    return jsonify({
        "status": "success",
        "timeline": timeline
    })

@app.route('/api/dashboard/pending-actions', methods=['GET'])
def pending_actions():
    """Get a list of trades that require attention or action"""
    # Get trades with statuses requiring attention
    failed_validation = workflow.get_trades_by_status(TradeStatus.VALIDATION_FAILED.value)
    failed_clearing = workflow.get_trades_by_status(TradeStatus.CLEARING_FAILED.value)
    failed_settlement = workflow.get_trades_by_status(TradeStatus.FAILED.value)
    
    # Combine and sort by most recent first
    all_attention_required = failed_validation + failed_clearing + failed_settlement
    all_attention_required = sorted(all_attention_required, key=lambda x: x['updated_at'], reverse=True)
    
    return jsonify({
        "status": "success",
        "count": len(all_attention_required),
        "trades_requiring_attention": all_attention_required
    })

@app.route('/api/dashboard/trading-volume', methods=['GET'])
def trading_volume():
    """Get trading volume data for charting"""
    # Get all trades
    all_trades = workflow.get_all_trades()
    
    # Organize by date
    volume_by_date = {}
    
    for trade in all_trades:
        trade_date = trade.get('trade_date')
        if trade_date not in volume_by_date:
            volume_by_date[trade_date] = {
                "count": 0,
                "value": 0.0,
                "securities": {}
            }
        
        # Update counts and values
        volume_by_date[trade_date]["count"] += 1
        trade_value = float(trade['quantity']) * float(trade['price'])
        volume_by_date[trade_date]["value"] += trade_value
        
        # Track by security
        security_id = trade.get('security_id')
        if security_id not in volume_by_date[trade_date]["securities"]:
            volume_by_date[trade_date]["securities"][security_id] = {
                "count": 0,
                "value": 0.0
            }
        
        volume_by_date[trade_date]["securities"][security_id]["count"] += 1
        volume_by_date[trade_date]["securities"][security_id]["value"] += trade_value
    
    # Sort by date
    sorted_dates = sorted(volume_by_date.keys())
    result = [{"date": date, **volume_by_date[date]} for date in sorted_dates]
    
    return jsonify({
        "status": "success",
        "trading_volume": result
    })

# WebSocket event handling
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Use SocketIO instead of regular Flask run
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)