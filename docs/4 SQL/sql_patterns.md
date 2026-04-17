---
icon: lucide/file-code-corner
---

# SQL Patterns

## Sessionization (Session Gap)

`Identify new user seesions where inactivity exceeds 30 minutes.`

**Sample Input**

|user_id|event_time|
|-------|----------|
|U-1045|2023-10-01 10:00|
|U-1045|2023-10-01 10:15|
|U-1045|2023-10-01 11:10|


**Sample Output**

|user_id|event_time|session_id|
|-------|----------|----------|
|U-1045|2023-10-01 10:00|1|
|U-1045|2023-10-01 10:15|1|
|U-1045|2023-10-01 11:10|2|

```sql
SELECT user_id, event_time,
SUM(is_new) OVER (PARTITION BY user_id ORDER BY event_time) as session_id
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