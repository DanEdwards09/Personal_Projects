"""
Simple REST API for the Trade Settlement Workflow using Flask
"""

from flask import Flask, request, jsonify, render_template
from simple_trade_workflow import SimpleTradeWorkflow, TradeStatus

app = Flask(__name__)
workflow = SimpleTradeWorkflow()

@app.route('/')
def index():
    """Simple homepage with links to API documentation"""
    return """
    <h1>Simple Trade Settlement API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><a href="/trades">/trades</a> - GET all trades or POST to create a new trade</li>
        <li><a href="/trades/status/NEW">/trades/status/{status}</a> - GET trades by status</li>
        <li>/trades/{trade_id} - GET details for a specific trade</li>
        <li>/trades/{trade_id}/history - GET history for a specific trade</li>
        <li>/trades/{trade_id}/cancel - POST to cancel a trade</li>
    </ul>
    """

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
    
    result = workflow.cancel_trade(trade_id, reason)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify(result), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)