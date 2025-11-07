// static/js/dashboard.js

// Wait for document to load
document.addEventListener('DOMContentLoaded', function() {
  const dashboardRoot = document.getElementById('dashboard-root');
  
  // Function to fetch dashboard data
  async function fetchDashboardData() {
    try {
      // Fetch summary data
      const summaryResponse = await fetch('/api/dashboard/summary');
      const summaryData = await summaryResponse.json();
      
      // Fetch timeline data
      const timelineResponse = await fetch('/api/dashboard/settlement-timeline');
      const timelineData = await timelineResponse.json();
      
      // Update the dashboard with the data
      updateDashboard(summaryData, timelineData.timeline);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      dashboardRoot.innerHTML = `<div class="bg-red-100 p-4 rounded">
        <p class="text-red-700">Error loading dashboard data. Please try again.</p>
      </div>`;
    }
  }
  
  // Function to update the dashboard with data
  function updateDashboard(summary, timeline) {
    // Create simple HTML representation of the dashboard
    let html = `
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-white p-4 rounded shadow">
          <h2 class="text-gray-500 text-sm mb-1">Total Trades</h2>
          <p class="text-2xl font-bold">${summary.total_trades || 0}</p>
        </div>
        <div class="bg-white p-4 rounded shadow">
          <h2 class="text-gray-500 text-sm mb-1">Settled Trades</h2>
          <p class="text-2xl font-bold text-green-600">${summary.status_counts?.SETTLED || 0}</p>
        </div>
        <div class="bg-white p-4 rounded shadow">
          <h2 class="text-gray-500 text-sm mb-1">Failed Trades</h2>
          <p class="text-2xl font-bold text-red-600">${
            (summary.status_counts?.FAILED || 0) + 
            (summary.status_counts?.VALIDATION_FAILED || 0) + 
            (summary.status_counts?.CLEARING_FAILED || 0)
          }</p>
        </div>
        <div class="bg-white p-4 rounded shadow">
          <h2 class="text-gray-500 text-sm mb-1">Pending Settlement</h2>
          <p class="text-2xl font-bold text-blue-600">${
            (summary.status_counts?.VALIDATED || 0) + 
            (summary.status_counts?.CLEARED || 0) + 
            (summary.status_counts?.MATCHED || 0)
          }</p>
        </div>
      </div>
    `;
    
    // Add failed trades section
    html += `
      <div class="bg-white p-4 rounded shadow mb-6">
        <h2 class="text-lg font-bold mb-4">Recent Failed Trades</h2>
    `;
    
    if (summary.recent_failed_trades?.length > 0) {
      html += `
        <table class="min-w-full bg-white">
          <thead>
            <tr>
              <th class="px-4 py-2 text-left text-sm font-medium text-gray-500">Trade ID</th>
              <th class="px-4 py-2 text-left text-sm font-medium text-gray-500">Security</th>
              <th class="px-4 py-2 text-left text-sm font-medium text-gray-500">Quantity</th>
              <th class="px-4 py-2 text-left text-sm font-medium text-gray-500">Failure Reason</th>
            </tr>
          </thead>
          <tbody>
      `;
      
      summary.recent_failed_trades.forEach(trade => {
        html += `
          <tr class="border-t hover:bg-gray-50">
            <td class="px-4 py-2 text-sm">${trade.trade_id}</td>
            <td class="px-4 py-2 text-sm">${trade.security_id}</td>
            <td class="px-4 py-2 text-sm">${trade.quantity}</td>
            <td class="px-4 py-2 text-sm text-red-600">${trade.notes}</td>
          </tr>
        `;
      });
      
      html += `
          </tbody>
        </table>
      `;
    } else {
      html += `
        <div class="text-center py-4">
          <p>No failed trades found!</p>
        </div>
      `;
    }
    
    html += `</div>`;
    
    // Add timeline section
    html += `
      <div class="bg-white p-4 rounded shadow mb-6">
        <h2 class="text-lg font-bold mb-4">Upcoming Settlements</h2>
        <table class="min-w-full">
          <thead>
            <tr>
              <th class="px-4 py-2 text-left">Date</th>
              <th class="px-4 py-2 text-left">Number of Trades</th>
              <th class="px-4 py-2 text-left">Total Value</th>
            </tr>
          </thead>
          <tbody>
    `;
    
    Object.entries(timeline).forEach(([date, data]) => {
      html += `
        <tr class="border-t">
          <td class="px-4 py-2">${date}</td>
          <td class="px-4 py-2">${data.count}</td>
          <td class="px-4 py-2">$${data.value.toLocaleString()}</td>
        </tr>
      `;
    });
    
    html += `
          </tbody>
        </table>
      </div>
    `;
    
    // Update the dashboard
    dashboardRoot.innerHTML = html;
    document.getElementById('last-updated').textContent = new Date().toLocaleString();
  }
  
  // Initial data fetch
  fetchDashboardData();
  
  // Set up WebSocket for real-time updates
  const socket = io();
  
  socket.on('connect', function() {
    console.log('Connected to WebSocket');
  });
  
  socket.on('trade_update', function(data) {
    console.log('Trade update received:', data);
    // Refresh dashboard data when a trade is updated
    fetchDashboardData();
  });
});