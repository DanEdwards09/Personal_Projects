CREATE TABLE cust_acct AS
SELECT 
c.*, 
a.account_id,
a.account_type,
a.open_date
FROM customers c 
LEFT JOIN accounts a 
ON c.customer_id = a.customer_id;