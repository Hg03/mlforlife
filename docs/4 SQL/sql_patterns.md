---
icon: lucide/file-code-corner
---

# SQL Patterns

## 1. Sessionization (Session Gap)
`Identify new user sessions when inactivity exceeds 30 minutes.`

**Sample Input**

| user_id | event_time |
|---- |---- |
| U-1045 | 2023-10-01 10:00 |
| U-1045 | 2023-10-01 10:15 |
| U-1045 | 2023-10-01 11:10 |

**Sample Output**

| user_id | event_time | session_id |
|---- |---- |---- |
| U-1045 | 2023-10-01 10:00 | 1 |
| U-1045 | 2023-10-01 10:15 | 1 |
| U-1045 | 2023-10-01 11:10 | 2 |

**Query**
```sql
SELECT user_id, event_time,
SUM(is_new) OVER (PARTITION BY user_id ORDER BY event_time) AS session_id
FROM (
    SELECT user_id, event_time,
    CASE
        WHEN DATEDIFF(minute, LAG(event_time) OVER
        (PARTITION BY user_id ORDER BY event_time), event_time) > 30
        THEN 1
        ELSE 0
    END AS is_new
    FROM events
) t;
```

---

## 2. LAG / LEAD (Growth Trend)
`Find customers whose spend increases month over month.`

**Sample Input**

| customer_id | month | amount |
|---- |---- |---- |
| C-992 | 2023-01 | 50.00 |
| C-992 | 2023-02 | 120.00 |
| C-114 | 2023-01 | 80.00 |
| C-114 | 2023-02 | 70.00 |

**Sample Output**

| customer_id |
|---- |
| C-992 |

**Query**
```sql
SELECT DISTINCT customer_id
FROM (
    SELECT customer_id, month, amount,
    LAG(amount) OVER (PARTITION BY customer_id ORDER BY month) AS prev_amt
    FROM spend
) t
WHERE amount > prev_amt;
```

---

## 3. Window Filter (Back-to-Back)
`Find users who placed orders on consecutive days.`

**Sample Input**

| customer_id | order_date |
|---- |---- |
| C-105 | 2023-11-03 |
| C-105 | 2023-11-04 |
| C-202 | 2023-11-10 |

**Sample Output**

| customer_id |
|---- |
| C-105 |

**Query**
```sql
SELECT DISTINCT customer_id
FROM (
    SELECT customer_id, order_date,
    LAG(order_date) OVER
    (PARTITION BY customer_id ORDER BY order_date) AS prev_date
    FROM orders
) t
WHERE DATEDIFF(day, prev_date, order_date) = 1;
```

---

## 4. NOT IN (Churned Users)
`Find customers active last month but not this month.`

**Sample Input**

| customer_id | activity_date |
|---- |---- |
| U-801 | 2024-02-15 |
| U-801 | 2024-03-05 |
| U-802 | 2024-02-20 |

**Sample Output**

| customer_id |
|---- |
| U-802 |

**Query**
```sql
SELECT DISTINCT a.customer_id
FROM activity a
WHERE MONTH(a.activity_date) = MONTH(CURDATE())
AND YEAR(a.activity_date) = YEAR(CURDATE())
AND a.customer_id NOT IN (
    SELECT customer_id
    FROM activity
    WHERE MONTH(activity_date) = MONTH(DATE_ADD(CURDATE(), INTERVAL 1 MONTH))
    AND YEAR(activity_date) = YEAR(DATE_ADD(CURDATE(), INTERVAL 1 MONTH))
);
```

---

## 5. HAVING / COUNT (Loyal Users)
`Identify customers who placed orders in every available month.`

**Sample Input**

| customer_id | order_date |
|---- |---- |
| C-44 | 2023-01-10 |
| C-44 | 2023-02-15 |
| C-55 | 2023-01-20 |

**Sample Output**

| customer_id |
|---- |
| C-44 |

**Query**
```sql
SELECT customer_id
FROM orders
GROUP BY customer_id
HAVING COUNT(DISTINCT MONTH(order_date)) =
(SELECT COUNT(DISTINCT MONTH(order_date)) FROM orders);
```

---

## 6. LEFT JOIN (Missing IDs)
`Find missing order IDs in a sequential numeric list.`

**Sample Input**

| id |
|---- |
| 1001 |
| 1002 |
| 1004 |
| 1005 |

**Sample Output**

| missing_id |
|---- |
| 1003 |

**Query**
```sql
SELECT o1.id + 1 AS missing_id
FROM orders o1
LEFT JOIN orders o2 ON o1.id + 1 = o2.id
WHERE o2.id IS NULL
AND o1.id < (SELECT MAX(id) FROM orders);
```

---

## 7. Gaps & Islands (3-Day Streak)
`Find users active for at least 3 consecutive days.`

**Sample Input**

| user_id | login_date |
|---- |---- |
| U-77 | 2023-10-01 |
| U-77 | 2023-10-02 |
| U-77 | 2023-10-03 |
| U-88 | 2023-10-01 |

**Sample Output**

| user_id |
|---- |
| U-77 |

**Query**
```sql
SELECT user_id
FROM (
    SELECT user_id, login_date,
    DATEADD(day, -ROW_NUMBER() OVER
    (PARTITION BY user_id ORDER BY login_date), login_date) AS grp
    FROM logins
) t
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

---

## 8. Window Ratio (Share of Total)
`Calculate category-wise percentage distribution of revenue.`

**Sample Input**

| category | revenue |
|---- |---- |
| Phones | 50000 |
| Laptops | 150000 |

**Sample Output**

| category | pct |
|---- |---- |
| Phones | 25.0 |
| Laptops | 75.0 |

**Query**
```sql
SELECT
category,
SUM(revenue) * 100.0 / SUM(SUM(revenue)) OVER () AS pct
FROM sales
GROUP BY category;
```

---

## 9. CASE WHEN (Delivery SLA)
`Categorize orders based on delivery duration (Fast <= 2 Days).`

**Sample Input**

| order_id | order_date | delivery_date |
|---- |---- |---- |
| ORD-1 | 2023-12-01 | 2023-12-02 |
| ORD-2 | 2023-12-01 | 2023-12-05 |

**Sample Output**

| order_id | delivery_type |
|---- |---- |
| ORD-1 | Fast |
| ORD-2 | Slow |

**Query**
```sql
SELECT
order_id,
CASE
    WHEN DATEDIFF(day, order_date, delivery_date) <= 2
    THEN 'Fast'
    ELSE 'Slow'
END AS delivery_type
FROM orders;
```

---

## 10. Self Join (Org Chart)
`Fetch employee, manager, and senior manager names.`

**Sample Input**

| name | manager_id | id |
|---- |---- |---- |
| Bob Smith | 102 | 101 |
| Alice Lee | 103 | 102 |
| Sarah CEO | NULL | 103 |

**Sample Output**

| emp_name | mgr_name | sr_mgr_name |
|---- |---- |---- |
| Bob Smith | Alice Lee | Sarah CEO |

**Query**
```sql
SELECT
e.name AS emp_name,
m.name AS mgr_name,
sm.name AS sr_mgr_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id
LEFT JOIN employees sm ON m.manager_id = sm.id;
```

---

## 11. NOT EXISTS (Failed Payments)
`Find orders that have no matching payments.`

**Sample Input**

| id | total_amount |
|---- |---- |
| ORD-99 | 100.00 |
| ORD-100 | 50.00 |

**Sample Output**

| id |
|---- |
| ORD-100 |

**Query**
```sql
SELECT o.id
FROM orders o
WHERE NOT EXISTS (
    SELECT 1 FROM payments p
    WHERE p.order_id = o.id
);
```

---

## 12. Manual Pivot (Row to Column)
`Convert row-based monthly sales into aggregated columns.`

**Sample Input**

| month | sales |
|---- |---- |
| Jan | 15000 |
| Feb | 25000 |
| Mar | 10000 |

**Sample Output**

| Jan | Feb | Mar |
|---- |---- |---- |
| 15000 | 25000 | 10000 |

**Query**
```sql
SELECT
SUM(CASE WHEN month = 'Jan' THEN sales END) AS Jan,
SUM(CASE WHEN month = 'Feb' THEN sales END) AS Feb,
SUM(CASE WHEN month = 'Mar' THEN sales END) AS Mar
FROM sales;
```

---

## 13. Theta Join (Double Booking)
`Identify overlapping booking intervals for properties.`

**Sample Input**

| id | start_date | end_date |
|---- |---- |---- |
| B-10 | 2023-06-01 | 2023-06-05 |
| B-11 | 2023-06-04 | 2023-06-10 |

**Sample Output**

| id | start_date | end_date |
|---- |---- |---- |
| B-10 | 2023-06-01 | 2023-06-05 |
| B-11 | 2023-06-04 | 2023-06-10 |

**Query**
```sql
SELECT a.*
FROM bookings a
JOIN bookings b
ON a.id != b.id
AND a.start_date < b.end_date
AND a.end_date > b.start_date;
```

---

## 14. Math Window (True Middle)
`Calculate the true mathematical median of employee salaries.`

**Sample Input**

| salary |
|---- |
| 60000 |
| 70000 |
| 75000 |
| 80000 |
| 120000 |

**Sample Output**

| median |
|---- |
| 75000 |

**Query**
```sql
SELECT AVG(salary) AS median
FROM (
    SELECT salary,
    ROW_NUMBER() OVER (ORDER BY salary) AS rn,
    COUNT(*) OVER () AS cnt
    FROM employees
) t
WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2);
```

---

## 15. Group Filter (2-Month Active)
`Find users active in both Month 1 and Month 2.`

**Sample Input**

| user_id | month |
|---- |---- |
| U-500 | 2024-01 |
| U-500 | 2024-02 |
| U-600 | 2024-01 |

**Sample Output**

| user_id |
|---- |
| U-500 |

**Query**
```sql
SELECT DISTINCT user_id
FROM activity
WHERE month IN ('2024-01', '2024-02')
GROUP BY user_id
HAVING COUNT(DISTINCT month) = 2;
```

---

## 16. Adv Window (Cross-Dept Avg)
`Find employees earning more than the company avg (excluding their own dept).`

**Sample Input**

| emp_name | dept | salary |
|---- |---- |---- |
| John | Sales | 120000 |
| Jane | IT | 80000 |
| Mike | IT | 60000 |

**Sample Output**

| emp_name |
|---- |
| John |

**Query**
```sql
SELECT emp_name FROM (
    SELECT emp_name, salary,
    (SUM(salary) OVER() - SUM(salary) OVER(PARTITION BY dept)) /
    NULLIF(COUNT(*) OVER() - COUNT(*) OVER(PARTITION BY dept), 0) AS out_avg
    FROM employees
) t
WHERE salary > out_avg;
```

---

## 17. Running Total (Wallet Balance)
`Calculate final user balance from credits (CR) and debits (DB).`

**Sample Input**

| user_id | type | amount |
|---- |---- |---- |
| U-123 | CR | 150.00 |
| U-123 | DB | 45.00 |
| U-123 | CR | 20.00 |

**Sample Output**

| user_id | balance |
|---- |---- |
| U-123 | 125.00 |

**Query**
```sql
SELECT
user_id,
SUM(CASE WHEN type = 'CR' THEN amount ELSE -amount END) AS balance
FROM transactions
GROUP BY user_id;
```

---

## 18. DENSE_RANK (Top 2 Salaries)
`Find the top 2 highest paid employees per department.`

**Sample Input**

| dept | salary |
|---- |---- |
| Eng | 150000 |
| Eng | 150000 |
| Eng | 140000 |
| Eng | 120000 |

**Sample Output**

| dept | salary |
|---- |---- |
| Eng | 150000 |
| Eng | 140000 |

**Query**
```sql
SELECT dept, salary FROM (
    SELECT dept, salary,
    DENSE_RANK() OVER(PARTITION BY dept ORDER BY salary DESC) AS rnk
    FROM employees
) t
WHERE rnk <= 2;
```

---

## 19. Rolling Sum (7-Day Avg)
`Calculate a 7-day rolling average of daily revenue.`

**Sample Input**

| date | revenue |
|---- |---- |
| 2023-11-01 | 1000 |
| 2023-11-02 | 2000 |
| 2023-11-03 | 3000 |

**Sample Output**

| date | rolling_avg |
|---- |---- |
| 2023-11-01 | 1000.0 |
| 2023-11-02 | 1500.0 |
| 2023-11-03 | 2000.0 |

**Query**
```sql
SELECT
date,
AVG(revenue) OVER(
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
) AS rolling_avg
FROM daily_sales;
```

---

## 20. Checklist
`Summary of the 20 mastered SQL patterns.`

* **Sessionization**
* **Lag / Lead**
* **Window Filter**
* **Not In**
* **Having / Count**
* **Left Join**
* **Gaps & Islands**
* **Window Ratio**
* **Case When**
* **Self Join**
* **Not Exists**
* **Manual Pivot**
* **Theta Join**
* **Math Window**
* **Group Filter**
* **Adv Window**
* **Running Total**
* **Dense_Rank**
* **Rolling Sum**
* **Success Factors**