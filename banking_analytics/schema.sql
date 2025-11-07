-- fs_assessment_starter schema
CREATE TABLE customers (
  customer_id TEXT PRIMARY KEY,
  join_date DATE,
  region TEXT,
  age INTEGER,
  annual_income NUMERIC,
  segment TEXT
);

CREATE TABLE accounts (
  account_id TEXT PRIMARY KEY,
  customer_id TEXT REFERENCES customers(customer_id),
  account_type TEXT,
  open_date DATE
);

CREATE TABLE transactions (
  transaction_id TEXT PRIMARY KEY,
  account_id TEXT REFERENCES accounts(account_id),
  txn_date DATE,
  category TEXT,
  channel TEXT,
  amount NUMERIC
);

CREATE TABLE chargebacks (
  chargeback_id TEXT PRIMARY KEY,
  account_id TEXT REFERENCES accounts(account_id),
  transaction_id TEXT REFERENCES transactions(transaction_id),
  cb_date DATE,
  cb_amount NUMERIC,
  reason TEXT
);

CREATE TABLE monthly_balances (
  account_id TEXT REFERENCES accounts(account_id),
  ym TEXT,
  ending_balance NUMERIC
);

CREATE TABLE customer_churn (
  customer_id TEXT REFERENCES customers(customer_id),
  last_txn_date DATE,
  churned BOOLEAN
);